
"""This module provides advanced imaging techniques for thermal signal analysis, including phase coherence imaging, Hilbert transform analysis, thermal signal reconstruction, modulated thermography, principal component thermography, pulsed phase thermography, and wavelet transform analysis.
Functions:
    clear_gpu_memory(): Clear GPU memory by deleting arrays and synchronizing.
    get_max_chunk_frames(height, width, dtype, extra_arrays=2, overhead=0.9): Calculate the maximum number of frames that can be processed in a chunk based on available GPU memory.
    get_max_chunk_pixels(frames, dtype, extra_arrays=2, overhead=0.9): Calculate the maximum number of pixels that can be processed in a chunk based on available GPU memory.
    phase_coherence_imaging(T, fs, f_stim): Perform phase coherence imaging on the input signal.
    hilbert_transform_analysis(T): Perform Hilbert transform analysis on the input signal.
    thermal_signal_reconstruction(T, order=5): Reconstruct thermal signals using polynomial fitting.
    modulated_thermography(T, fs, f_stim, harmonics=[2, 3]): Perform modulated thermography on the input signal.
    principal_component_thermography(T, n_components=5): Perform principal component analysis on the input signal.
    pulsed_phase_thermography(T, fs): Perform pulsed phase thermography on the input signal.
    wavelet_transform_analysis(T, wavelet='db4', level=3): Perform wavelet transform analysis on the input signal.
"""

from atpp.logging_config import logger 
import tempfile
from tqdm import tqdm
import gc
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import os
from multiprocessing import cpu_count

# Retrieve number of processors automatically
NUM_PROCESSORS = os.cpu_count() or cpu_count()  # Fallback to multiprocessing if os.cpu_count() fails

# Try importing cupy for GPU-accelerated computing
try:
    import cupy as cp
    USE_GPU = True
    # Free memory
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()
    logger.info("Using GPU-accelerated computing")
except ImportError:
    USE_GPU = False
    cp = np  # Fallback to numpy if GPU is not available
    logger.info("Using CPU-based computing")

def clear_gpu_memory():
    """Clear GPU memory by deleting arrays and synchronizing."""
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()

def get_max_chunk_frames(height, width, dtype, extra_arrays=2, overhead=0.9):
    """Calculate the maximum number of frames that can be processed in a single chunk based on available GPU memory.

    :param height: The height of each frame.
    :type height: int
    :param width: The width of each frame.
    :type width: int
    :param dtype: The data type of the frame elements.
    :type dtype: numpy.dtype or cupy.dtype
    :param extra_arrays: The number of additional arrays to consider in memory calculation, defaults to 2.
    :param overhead: The fraction of free memory to use, defaults to 0.9.
    :return: The maximum number of frames that can be processed in a single chunk.
    :rtype: int or None
    """
    if USE_GPU:
        free_mem, total_mem = cp.cuda.Device().mem_info
        free_mem *= overhead  # Use only a fraction of the free memory

        bytes_per_element = cp.dtype(dtype).itemsize
        per_frame_memory = height * width * bytes_per_element * (1 + extra_arrays)

        max_frames = int(free_mem / per_frame_memory)
        max_frames = max(1, max_frames)  # Ensure at least one frame is processed
        return max_frames
    else:
        return None  # Not applicable for CPU processing

def get_max_chunk_pixels(frames, dtype, extra_arrays=2, overhead=0.9):
    """Calculate the maximum number of pixels that can be processed in a single chunk based on available GPU memory.

    :param frames: The number of frames.
    :type frames: int
    :param dtype: The data type of the frame elements.
    :type dtype: numpy.dtype or cupy.dtype
    :param extra_arrays: The number of additional arrays to consider in memory calculation, defaults to 2.
    :param overhead: The fraction of free memory to use, defaults to 0.9.
    :type extra_arrays: int, optional
    :type overhead: float, optional
    :return: The maximum number of pixels that can be processed in a single chunk.
    :rtype: int or None
    """
    if USE_GPU:
        # Synchronize and clear the CuPy memory pool
        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks()

        # Get the correct memory information
        free_mem, total_mem = cp.cuda.Device().mem_info
        free_mem *= overhead  # Use only a fraction of the free memory

        bytes_per_element = cp.dtype(dtype).itemsize
        per_pixel_memory = frames * bytes_per_element * (1 + extra_arrays)

        max_pixels = int(free_mem / per_pixel_memory)
        max_pixels = max(1, max_pixels)  # Ensure at least one pixel is processed
        return max_pixels
    else:
        return None  # Not applicable for CPU processing

def phase_coherence_imaging(T, fs, f_stim):
    """Perform phase coherence imaging on the input signal.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray or cupy.ndarray
    :param fs: Sampling frequency of the input signal.
    :type fs: float
    :param f_stim: Stimulation frequency for phase coherence imaging.
    :type f_stim: float
    :return: Tuple containing amplitude, phase, and phase coherence arrays.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    try:
        if USE_GPU:
            logger.info("Starting phase coherence imaging on GPU")
            # Convert T to CuPy array for GPU processing
            height, width, frames = T.shape

            # Calculate the maximum number of pixels we can process at once
            max_chunk_pixels = get_max_chunk_pixels(frames, cp.float16, extra_arrays=2, overhead=0.9)
            total_pixels = height * width

            # Initialize I and Q arrays
            I = np.zeros((height, width), dtype=np.float16)
            Q = np.zeros((height, width), dtype=np.float16)

            # Process data in chunks
            for start_pixel in tqdm(range(0, total_pixels, max_chunk_pixels), desc="Processing chunks"):
                end_pixel = min(start_pixel + max_chunk_pixels, total_pixels)
                chunk_indices = np.unravel_index(range(start_pixel, end_pixel), (height, width))

                # Extract the chunk of data
                T_chunk = T[chunk_indices]
                T_chunk_gpu = cp.asarray(T_chunk, dtype=cp.float16)

                # Create time vector
                t = cp.arange(frames) / fs

                # Reference signals for demodulation
                ref_sin = cp.sin(2 * cp.pi * f_stim * t)
                ref_cos = cp.cos(2 * cp.pi * f_stim * t)

                # Reshape reference signals for broadcasting
                ref_sin = ref_sin[cp.newaxis, :]
                ref_cos = ref_cos[cp.newaxis, :]

                # Compute I and Q using vectorized operations on GPU
                I_chunk_gpu = cp.sum(T_chunk_gpu * ref_cos, axis=1) / frames
                Q_chunk_gpu = cp.sum(T_chunk_gpu * ref_sin, axis=1) / frames

                # Transfer results back to CPU
                I_chunk = I_chunk_gpu.get()
                Q_chunk = Q_chunk_gpu.get()

                # Assign to the appropriate locations in the I and Q arrays
                I[chunk_indices] = I_chunk
                Q[chunk_indices] = Q_chunk

                # Free GPU memory
                del T_chunk_gpu, I_chunk_gpu, Q_chunk_gpu
                clear_gpu_memory()

            # Compute amplitude and phase
            amplitude = np.sqrt(I**2 + Q**2)
            phase = np.arctan2(Q, I)

            # Phase coherence computation
            phase_diff = np.zeros((height, width), dtype=np.float16)
            for i in tqdm(range(1, height - 1), desc="Computing phase coherence"):
                for j in range(1, width - 1):
                    neighbors = [
                        phase[i - 1, j],
                        phase[i + 1, j],
                        phase[i, j - 1],
                        phase[i, j + 1]
                    ]
                    phase_diff[i, j] = np.std([phase[i, j] - neighbor for neighbor in neighbors])

            # Avoid division by zero
            max_diff = np.max(phase_diff)
            if (max_diff == 0):
                phase_coherence = np.ones((height, width), dtype=np.float16)
            else:
                phase_coherence = 1 - (phase_diff / max_diff)

            # Clear GPU memory
            clear_gpu_memory()
            logger.info("Phase coherence imaging on GPU completed")

        else:
            logger.info("Starting phase coherence imaging on CPU")
            # CPU-based processing
            height, width, frames = T.shape

            # Create time vector
            t = np.arange(frames) / fs

            # Reference signals for demodulation
            ref_sin = np.sin(2 * np.pi * f_stim * t)
            ref_cos = np.cos(2 * np.pi * f_stim * t)

            # Initialize I and Q arrays
            I = np.zeros((height, width), dtype=np.float16)
            Q = np.zeros((height, width), dtype=np.float16)

            def calculate_iq(i):
                """
                Calculate I and Q for a specific row.
                """
                row_I = np.sum(T[i, :, :] * ref_cos, axis=1) / frames
                row_Q = np.sum(T[i, :, :] * ref_sin, axis=1) / frames
                return i, row_I, row_Q

            # Use ProcessPoolExecutor to parallelize row computations
            with ProcessPoolExecutor(max_workers=NUM_PROCESSORS) as executor:
                for i, row_I, row_Q in tqdm(executor.map(calculate_iq, range(height)), total=height, desc="Calculating I and Q"):
                    I[i, :] = row_I
                    Q[i, :] = row_Q

            # Compute amplitude and phase
            amplitude = np.sqrt(I**2 + Q**2)
            phase = np.arctan2(Q, I)

            # Phase coherence computation
            phase_diff = np.zeros((height, width), dtype=np.float16)
            for i in tqdm(range(1, height - 1), desc="Computing phase coherence"):
                for j in range(1, width - 1):
                    neighbors = [
                        phase[i - 1, j],
                        phase[i + 1, j],
                        phase[i, j - 1],
                        phase[i, j + 1],
                        phase[i - 1, j - 1],
                        phase[i - 1, j + 1],
                        phase[i + 1, j - 1],
                        phase[i + 1, j + 1]
                    ]
                    phase_diff[i, j] = np.std([phase[i, j] - neighbor for neighbor in neighbors])

            # Avoid division by zero
            max_diff = np.max(phase_diff)
            if max_diff == 0:
                phase_coherence = np.ones((height, width), dtype=np.float16)
            else:
                phase_coherence = 1 - (phase_diff / max_diff)

            logger.info("Phase coherence imaging on CPU completed")

        return amplitude, phase, phase_coherence

    except Exception as e:
        logger.error(f"Error in phase_coherence_imaging: {e}")
        return None, None, None

# def synchronous_demodulation(T, fs, f_stim):
#     """Perform synchronous demodulation on the input signal.

#         T (numpy.ndarray or cupy.ndarray): The input signal array with dimensions (height, width, frames).
#         fs (float): The sampling frequency of the input signal.
#         f_stim (float): The stimulation frequency for demodulation.

#         tuple: A tuple containing:
#             - amplitude (numpy.ndarray): The amplitude of the demodulated signal.
#             - phase (numpy.ndarray): The phase of the demodulated signal.

#     Raises:
#         Exception: If an error occurs during the demodulation process.
    
#     Returns:
#         tuple: A tuple containing:
#             - amplitude (numpy.ndarray): The amplitude of the demodulated signal.
#             - phase (numpy.ndarray): The phase of the demodulated signal.
#     """
#     try:
#         if USE_GPU:
#             logger.info("Starting synchronous demodulation on GPU")
#             if not isinstance(T, np.ndarray):
#                 raise TypeError(f"Expected T to be a numpy.ndarray, but got {type(T)}")
#             # Convert T to CuPy array for GPU processing
#             T_gpu = cp.asarray(T, dtype=cp.float16)
#             height, width, frames = T_gpu.shape

#             # Create time vector
#             t = cp.arange(frames) / fs

#             # Reference signals for demodulation
#             ref_sin = cp.sin(2 * cp.pi * f_stim * t)
#             ref_cos = cp.cos(2 * cp.pi * f_stim * t)

#             # Reshape reference signals for broadcasting
#             ref_sin = ref_sin[cp.newaxis, cp.newaxis, :]
#             ref_cos = ref_cos[cp.newaxis, cp.newaxis, :]

#             # Compute I and Q using vectorized operations on GPU
#             I_gpu = cp.sum(T_gpu * ref_cos, axis=2) / frames
#             Q_gpu = cp.sum(T_gpu * ref_sin, axis=2) / frames

#             # Transfer results back to CPU
#             I = I_gpu.get()
#             Q = Q_gpu.get()

#             # Compute amplitude and phase
#             amplitude = np.sqrt(I**2 + Q**2)
#             phase = np.arctan2(Q, I)

#             # Clear GPU memory
#             clear_gpu_memory()
#             logger.info("Synchronous demodulation on GPU completed")

#         else:
#             logger.info("Starting synchronous demodulation on CPU")
#             # CPU-based processing
#             height, width, frames = T.shape

#             # Create time vector
#             t = np.arange(frames) / fs

#             # Reference signals for demodulation
#             ref_sin = np.sin(2 * np.pi * f_stim * t)
#             ref_cos = np.cos(2 * np.pi * f_stim * t)

#             # Reshape reference signals for broadcasting
#             ref_sin = ref_sin[np.newaxis, np.newaxis, :]
#             ref_cos = ref_cos[np.newaxis, np.newaxis, :]

#             # Initialize I and Q arrays
#             I = np.zeros((height, width), dtype=np.float16)
#             Q = np.zeros((height, width), dtype=np.float16)

#             def calculate_iq(i):
#                 """
#                 Calculate I and Q for a specific row.
#                 """
#                 row_I = np.sum(T[i, :, :] * ref_cos, axis=1) / frames
#                 row_Q = np.sum(T[i, :, :] * ref_sin, axis=1) / frames
#                 return i, row_I, row_Q

#             # Use ProcessPoolExecutor to parallelize row computations
#             with ProcessPoolExecutor(max_workers=NUM_PROCESSORS) as executor:
#                 for i, row_I, row_Q in tqdm(executor.map(calculate_iq, range(height)), total=height, desc="Calculating I and Q"):
#                     I[i, :] = row_I
#                     Q[i, :] = row_Q

#             # Compute amplitude and phase
#             amplitude = np.sqrt(I**2 + Q**2)
#             phase = np.arctan2(Q, I)
#             logger.info("Synchronous demodulation on CPU completed")

#         return amplitude, phase

#     except Exception as e:
#         logger.error(f"Error in synchronous_demodulation: {e}")
#         return None, None

def hilbert_transform_analysis(T):
    """Perform Hilbert transform analysis on the input signal.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray or cupy.ndarray
    :return: Tuple containing amplitude and phase arrays.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    try:
        from scipy.signal import hilbert
        if USE_GPU:
            logger.info("Starting Hilbert transform analysis on GPU")
            height, width, frames = T.shape

            # Calculate the maximum number of pixels we can process at once
            max_chunk_pixels = get_max_chunk_pixels(frames, cp.complex128, extra_arrays=7, overhead=0.9)
            total_pixels = height * width
            logger.info(f"max_chunk_pixels: {max_chunk_pixels}, total_pixels: {total_pixels}")

            # Initialize amplitude and phase arrays
            amplitude = np.zeros((height, width, frames), dtype=np.float16)
            phase = np.zeros((height, width, frames), dtype=np.float16)

            rows_per_chunk = max_chunk_pixels // width
            row_chunks = (height) // rows_per_chunk

            if row_chunks != 1 and row_chunks != 0:
                logger.info(f"Needed chunks: {row_chunks}")
            elif row_chunks == 0:
                row_chunks = 1
                logger.info(f"Correcting --> Needed chunks: {row_chunks}")

            for chunk_idx in tqdm(range(row_chunks), desc="Processing chunks"):
                start_row = chunk_idx * rows_per_chunk
                end_row = min((chunk_idx + 1) * rows_per_chunk, height)  # Ensure we do not exceed the image height

                # Slicing the image arrays directly
                chunk_indices = np.s_[start_row:end_row, :]
                # Extract the chunk of data
                T_chunk = T[chunk_indices]
                T_chunk_gpu = cp.asarray(T_chunk, dtype=cp.float16)

                # Perform FFT-based Hilbert transform along the time axis (axis=2)
                fft_data = cp.fft.fft(T_chunk_gpu, axis=2)

                # Create the Hilbert transform multiplier
                frames_chunk = frames
                h = cp.zeros(frames_chunk, dtype=cp.float16)
                if frames_chunk % 2 == 0:
                    h[0] = 1
                    h[1:frames_chunk // 2] = 2
                    h[frames_chunk // 2] = 1
                else:
                    h[0] = 1
                    h[1:(frames_chunk + 1) // 2] = 2

                # Reshape h for broadcasting over the first two dimensions
                h = h[cp.newaxis, cp.newaxis, :]

                # Apply the Hilbert multiplier
                analytic_fft = fft_data * h

                # Inverse FFT along the time axis (axis=2) to get the analytic signal
                analytic_signal = cp.fft.ifft(analytic_fft, axis=2)

                # Extract amplitude and phase
                amplitude_gpu = cp.abs(analytic_signal)
                phase_gpu = cp.unwrap(cp.angle(analytic_signal))

                # Transfer results back to CPU
                amplitude_chunk = amplitude_gpu.get()
                phase_chunk = phase_gpu.get()

                # Assign to the appropriate locations in the amplitude and phase arrays
                amplitude[chunk_indices] = amplitude_chunk
                phase[chunk_indices] = phase_chunk

                # Free GPU memory
                del T_chunk_gpu, fft_data, analytic_fft, analytic_signal, amplitude_gpu, phase_gpu
                clear_gpu_memory()

            # Clear GPU memory
            clear_gpu_memory()
            logger.info("Hilbert transform analysis on GPU completed")

        else:
            logger.info("Starting Hilbert transform analysis on CPU")
            height, width, frames = T.shape

            # Initialize amplitude and phase arrays
            amplitude = np.zeros((height, width), dtype=np.float16)
            phase = np.zeros((height, width), dtype=np.float16)

            def calculate_hilbert(i):
                row_amplitude = np.zeros(width, dtype=np.float16)
                row_phase = np.zeros(width, dtype=np.float16)
                for j in range(width):
                    signal = T[i, j, :]
                    analytic_signal = hilbert(signal)
                    row_amplitude[j] = np.abs(analytic_signal)
                    row_phase[j] = np.unwrap(np.angle(analytic_signal))
                return i, row_amplitude, row_phase

            # Use ProcessPoolExecutor to parallelize row computations
            with ProcessPoolExecutor(max_workers=NUM_PROCESSORS) as executor:
                for i, row_amp, row_ph in tqdm(executor.map(calculate_hilbert, range(height)), total=height, desc="Calculating Hilbert transform"):
                    amplitude[i, :] = row_amp
                    phase[i, :] = row_ph
            logger.info("Hilbert transform analysis on CPU completed")

        return amplitude, phase

    except Exception as e:
        logger.error(f"Error in hilbert_transform_analysis: {e}")
        return None, None

def thermal_signal_reconstruction(T, order=5):
    """Reconstruct thermal signals using polynomial fitting. STILL TO BE TESTED"""
    height, width, frames = T.shape
    log_time = np.log(np.arange(1, frames + 1))
    T_reconstructed = np.zeros_like(T)

    def reconstruct_signal(i):
        for j in tqdm(range(width), desc=f"Processing row {i}"):
            signal = T[i, j, :]
            log_signal = np.log(signal + np.finfo(float).eps)
            coeffs = np.polyfit(log_time, log_signal, order)
            log_signal_fit = np.polyval(coeffs, log_time)
            T_reconstructed[i, j, :] = np.exp(log_signal_fit)

    with ProcessPoolExecutor(max_workers=NUM_PROCESSORS) as executor:
        list(tqdm(executor.map(reconstruct_signal, range(height)), total=height, desc="Reconstructing signals"))
    logger.info("Thermal signal reconstruction completed")

    return T_reconstructed

def modulated_thermography(T, fs, f_stim, harmonics=[2, 3]):
    """Perform modulated thermography on a given temperature dataset.

    This function demodulates the temperature data at specified harmonic frequencies
    to extract amplitude and phase information.

    :param T: Temperature data array with dimensions (height, width, frames).
    :type T: numpy.ndarray or cupy.ndarray
    :param fs: Sampling frequency of the temperature data.
    :type fs: float
    :param f_stim: Stimulation frequency used during the thermography.
    :type f_stim: float
    :param harmonics: List of harmonic frequencies to demodulate, defaults to [2, 3].
    :return: A tuple containing two dictionaries:
             - amplitude: Dictionary with harmonic frequencies as keys and corresponding amplitude arrays as values.
             - phase: Dictionary with harmonic frequencies as keys and corresponding phase arrays as values.
    :rtype: Tuple[Dict[int, numpy.ndarray], Dict[int, numpy.ndarray]]
    """
    logger.info("Starting modulated thermography")
    # Convert T to cupy array if GPU is used
    T_gpu = cp.asarray(T) if USE_GPU else T

    height, width, frames = T_gpu.shape
    amplitude = {}
    phase = {}

    def demodulate_harmonic(h):
        frame_range = cp.arange(frames) if USE_GPU else np.arange(frames)

        ref_sin = cp.sin(2 * cp.pi * f_stim * h * frame_range / fs) if USE_GPU else np.sin(2 * np.pi * f_stim * h * frame_range / fs)
        ref_cos = cp.cos(2 * cp.pi * f_stim * h * frame_range / fs) if USE_GPU else np.cos(2 * np.pi * f_stim * h * frame_range / fs)

        I = cp.zeros((height, width)) if USE_GPU else np.zeros((height, width))
        Q = cp.zeros((height, width)) if USE_GPU else np.zeros((height, width))

        def calculate_iq(i):
            for j in range(width):
                signal = T_gpu[i, j, :]
                I[i, j] = cp.sum(signal * ref_cos) / frames if USE_GPU else np.sum(signal * ref_cos) / frames
                Q[i, j] = cp.sum(signal * ref_sin) / frames if USE_GPU else np.sum(signal * ref_sin) / frames

        # Use ProcessPoolExecutor for parallel processing if working with CPU
        if not USE_GPU:
            with ProcessPoolExecutor(max_workers=NUM_PROCESSORS) as executor:
                list(tqdm(executor.map(calculate_iq, range(height)), total=height, desc=f"Demodulating harmonic {h}"))
        else:
            for i in tqdm(range(height), desc=f"Demodulating harmonic {h}"):
                calculate_iq(i)

        amplitude[h] = cp.sqrt(I**2 + Q**2) if USE_GPU else np.sqrt(I**2 + Q**2)
        phase[h] = cp.arctan2(Q, I) if USE_GPU else np.arctan2(Q, I)

    for h in harmonics:
        demodulate_harmonic(h)

    # Convert results back to numpy arrays if using GPU
    amplitude, phase = ({h: cp.asnumpy(amplitude[h]) for h in harmonics} if USE_GPU else amplitude,
                        {h: cp.asnumpy(phase[h]) for h in harmonics} if USE_GPU else phase)
    if USE_GPU:
        clear_gpu_memory()
    logger.info("Modulated thermography completed")
    return amplitude, phase

def principal_component_thermography(T, n_components=5):
    """Perform independent component thermography on the input signal.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param n_components: Number of independent components to extract, defaults to 5.
    :type n_components: int, optional
    :return: Reconstructed thermal signal array with independent components.
    :rtype: numpy.ndarray
    """
    from sklearn.decomposition import PCA
    logger.info("Starting principal component thermography")
    height, width, frames = T.shape
    data = T.reshape(-1, frames)
    data_mean = np.mean(data, axis=0)
    data_centered = data - data_mean
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_centered)
    pcs_images = [pc.reshape(height, width) for pc in principal_components.T]
    gc.collect()
    logger.info("Principal component thermography completed")
    return pcs_images

def pulsed_phase_thermography(T, fs):
    """Perform pulsed phase thermography on the input thermal signal.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray or cupy.ndarray
    :param fs: Sampling frequency of the input signal.
    :type fs: float
    :raises Exception: If an error occurs during the processing.
    :return: Tuple containing amplitude, phase, and frequency arrays.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    try:
        if USE_GPU:
            logger.info("Starting pulsed phase thermography on GPU")
            # Convert T to CuPy array for GPU processing
            T_gpu = cp.asarray(T, dtype=cp.float16)
            height, width, frames = T_gpu.shape
            logger.info(f"Processing on GPU with shape: {T_gpu.shape}")

            # Initialize list to collect FFT batches
            fft_data = []

            chunks = get_max_chunk_frames(height, width, cp.complex128, extra_arrays=7, overhead=0.9)

            # Process data in batches to manage GPU memory
            for start in tqdm(range(0, frames, chunks), desc="Processing frames"):
                end = min(start + chunks, frames)
                logger.info(f"Processing frames {start} to {end} on GPU")

                # Perform FFT on the current batch
                batch_fft = cp.fft.fft(T_gpu[:, :, start:end], axis=2)

                # Append the FFT result to the list
                fft_data.append(batch_fft.get())
                
                del batch_fft  

                # Optionally synchronize and free memory after each batch
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()


            logger.info(f"Concatenated FFT data shape: {fft_data.shape}")

            cp.get_default_memory_pool().free_all_blocks()

            # Get the frequencies corresponding to the FFT result
            freqs = cp.fft.fftfreq(frames, d=1/fs)
            pos_mask = freqs > 0

            # Apply the positive frequency mask
            fft_data = fft_data[:, :, pos_mask]
            freqs = freqs[pos_mask]
            logger.info(f"Positive frequencies count: {cp.sum(pos_mask)}")

            # Calculate amplitude and phase
            amplitude = cp.abs(fft_data)
            phase = cp.angle(fft_data)

            # Transfer results back to NumPy arrays
            amplitude = amplitude.get()
            phase = phase.get()
            freqs = freqs.get()

            # Clear GPU memory
            clear_gpu_memory()
            logger.info("Pulsed phase thermography on GPU completed")

        else:
            logger.info("Starting pulsed phase thermography on CPU")
            # Perform FFT on CPU using NumPy
            height, width, frames = T.shape
            logger.info(f"Processing on CPU with shape: {T.shape}")

            # Perform FFT along the time axis
            fft_data = np.fft.fft(T, axis=2)
            logger.info(f"FFT data shape: {fft_data.shape}")

            # Get the frequencies corresponding to the FFT result
            freqs = np.fft.fftfreq(frames, d=1/fs)
            pos_mask = freqs > 0
            logger.info(f"Positive frequencies count: {np.sum(pos_mask)}")

            # Apply the positive frequency mask
            fft_data = fft_data[:, :, pos_mask]
            freqs = freqs[pos_mask]

            # Calculate amplitude and phase
            amplitude = np.abs(fft_data)
            phase = np.angle(fft_data)
            logger.info("Pulsed phase thermography on CPU completed")

        return amplitude, phase, freqs

    except cp.cuda.memory.OutOfMemoryError as e:
        logger.error(f"GPU OutOfMemoryError: {e}")
        clear_gpu_memory()
        raise e
    except Exception as e:
        logger.error(f"Error in pulsed_phase_thermography: {e}")
        return None, None, None

def wavelet_transform_analysis(T, wavelet='db4', level=3):
    """Perform wavelet transform analysis on the input thermal signal.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param wavelet: Type of wavelet to use for the transform, defaults to 'db4'.
    :type wavelet: str, optional
    :param level: Decomposition level of the wavelet transform, defaults to 3.
    :type level: int, optional
    :return: List of wavelet coefficients for each pixel.
    :rtype: list
    """
    import pywt
    logger.info("Starting wavelet transform analysis")
    height, width, frames = T.shape
    coeffs = []

    def calculate_wavelet(i):
        for j in range(width):
            signal = T[i, j, :]
            coeff = pywt.wavedec(signal, wavelet, level=level)
            coeffs.append(coeff)

    with ProcessPoolExecutor(max_workers=NUM_PROCESSORS) as executor:
        list(tqdm(executor.map(calculate_wavelet, range(height)), total=height, desc="Calculating wavelet coefficients"))
    logger.info("Wavelet transform analysis completed")

    return coeffs

def visualize_comparison(T, fs, f_stim, time):
    """Visualize a comparison of different imaging techniques.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param fs: Sampling frequency of the input signal.
    :type fs: float
    :param f_stim: Stimulation frequency for the imaging techniques.
    :type f_stim: float
    :param time: Time vector corresponding to the frames.
    :type time: numpy.ndarray
    """
    logger.info("Starting visualization of comparison")
    models = {
        "Phase Coherence Imaging": phase_coherence_imaging(T, fs, f_stim),
        "Hilbert Transform": hilbert_transform_analysis(T),
        "Thermal Signal Reconstruction": thermal_signal_reconstruction(T),
        "Modulated Thermography": modulated_thermography(T, fs, f_stim),
        "Principal Component Thermography": principal_component_thermography(T),
        "Pulsed Phase Thermography": pulsed_phase_thermography(T, fs),
    }

    plt.figure(figsize=(20, 15))
    for idx, (title, result) in enumerate(models.items()):
        if isinstance(result, tuple):
            result = result[0]  # If tuple, get first output (like amplitude)
        plt.subplot(3, 3, idx + 1)
        plt.imshow(result, cmap="gray")
        plt.title(title)
        plt.colorbar()

    plt.tight_layout()
    plt.show()
    logger.info("Visualization completed")

def visualize_wavelet_coefficients(T, wavelet='db4', level=3):
    """
    STILL to be refactored
    Visualizes wavelet coefficients of a 3D thermal data array using scalograms.
    """
    import pywt
    logger.info("Starting visualization of wavelet coefficients")
    # Get dimensions
    height, width, frames = T.shape
    coeffs_list = []

    # Apply wavelet transform to each pixel time series
    for i in tqdm(range(height), desc="Processing rows"):
        for j in range(width):
            signal = T[i, j, :]
            coeffs = pywt.wavedec(signal, wavelet, level=level)
            coeffs_list.append(coeffs)

    # Visualize coefficients for a specific pixel (e.g., center pixel)
    center_index = height // 2, width // 2
    selected_coeffs = pywt.wavedec(T[center_index], wavelet, level=level)

    plt.figure(figsize=(15, 10))
    
    # Plot approximation coefficients
    plt.subplot(level + 1, 1, 1)
    plt.plot(selected_coeffs[0], label='Approximation Coefficients')
    plt.title(f'Approximation Coefficients - Level 0')
    plt.xlabel('Time')
    plt.ylabel('Coefficient Value')
    plt.legend()
    
    # Plot detail coefficients for each level
    for i, coeff in enumerate(selected_coeffs[1:], start=1):
        plt.subplot(level + 1, 1, i + 1)
        plt.plot(coeff, label=f'Detail Coefficients - Level {i}')
        plt.title(f'Detail Coefficients - Level {i}')
        plt.xlabel('Time')
        plt.ylabel('Coefficient Value')
        plt.legend()
        
    plt.tight_layout()
    plt.show()
    logger.info("Wavelet coefficients visualization completed")

def independent_component_thermography(T, n_components=5, pca_components=50):
    """Perform independent component thermography on the input signal with PCA pre-reduction.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param n_components: Number of independent components to extract, defaults to 5.
    :type n_components: int, optional
    :param pca_components: Number of PCA components to retain, defaults to 50.
    :type pca_components: int, optional
    :return: Reconstructed thermal signal array with independent components.
    :rtype: numpy.ndarray
    """
    try:
        from sklearn.decomposition import PCA, FastICA
        logger.info("Starting independent component thermography")
        height, width, frames = T.shape
        gc.collect()

        # Reshape data to (frames, pixels)
        data = T.reshape(-1, frames).astype(np.float32).T  # Shape: (frames, pixels)
        logger.info(f"Data reshaped to {data.shape} and converted to float32")

        del T
        gc.collect()

        # Perform PCA to reduce dimensionality
        pca = PCA(n_components=pca_components, random_state=0, svd_solver='randomized')
        data_pca = pca.fit_transform(data)  # Shape: (frames, pca_components)
        logger.info(f"PCA completed with {pca_components} components")
        del data
        gc.collect()

        # Perform ICA on PCA-reduced data
        ica = FastICA(n_components=n_components, random_state=0, max_iter=200, tol=0.0001)
        independent_components = ica.fit_transform(data_pca)  # Shape: (frames, n_components)
        logger.info("ICA completed")
        del data_pca
        gc.collect()

        # Reconstruct from ICA
        reconstructed_pca = ica.inverse_transform(independent_components)  # Shape: (frames, pca_components)
        reconstructed_data = pca.inverse_transform(reconstructed_pca)  # Shape: (frames, pixels)
        logger.info("Reconstruction completed")

        # Reshape reconstructed_data to original shape (height, width, frames)
        reconstructed_data = reconstructed_data.T.reshape(height, width, frames)
        return reconstructed_data

    except MemoryError:
        logger.error("MemoryError: The system ran out of memory during independent component thermography.")
        return None
    except Exception as e:
        logger.error(f"Error in independent_component_thermography: {e}")
        return None


def monogenic_signal_analysis(T):
    """Perform monogenic signal analysis on the input thermal signal.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :return: Tuple containing amplitude, phase, and orientation arrays.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)
    """
    try:
        logger.info("Starting monogenic signal analysis")
        import numpy as np
        from scipy import ndimage
        from scipy.fft import fftn, ifftn, fftshift

        # Get dimensions
        height, width, frames = T.shape
        
        # Compute the mean image over time
        T_mean = np.mean(T, axis=2)
        
        # Compute the Riesz transform kernels
        u = np.fft.fftfreq(height).reshape(-1, 1)
        v = np.fft.fftfreq(width).reshape(1, -1)
        radius = np.sqrt(u**2 + v**2) + np.finfo(float).eps  # Avoid division by zero

        # Riesz kernels
        R1 = -1j * u / radius
        R2 = -1j * v / radius

        # Perform Fourier transform of the mean image
        F = fftn(T_mean)

        # Apply Riesz transform
        R1F = R1 * F
        R2F = R2 * F

        # Inverse Fourier transform to get spatial domain representations
        monogenic_R1 = np.real(ifftn(R1F))
        monogenic_R2 = np.real(ifftn(R2F))

        # Compute local amplitude
        amplitude = np.sqrt(T_mean**2 + monogenic_R1**2 + monogenic_R2**2)

        # Compute local phase
        phase = np.arctan2(np.sqrt(monogenic_R1**2 + monogenic_R2**2), T_mean)

        # Compute local orientation
        orientation = np.arctan2(monogenic_R2, monogenic_R1)
        logger.info("Monogenic signal analysis completed")

        return amplitude, phase, orientation

    except Exception as e:
        logger.error(f"Error in monogenic_signal_analysis: {e}")
        return None, None, None

def phase_congruency_analysis(T, n_scale=4, n_orientation=4, min_wavelength=6, mult=2.1, sigma_onf=0.55, k=2.0, cut_off=0.5, g=10):
    """Perform phase congruency analysis on the input thermal signal.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param n_scale: Number of wavelet scales, defaults to 4.
    :type n_scale: int, optional
    :param n_orientation: Number of filter orientations, defaults to 4.
    :type n_orientation: int, optional
    :param min_wavelength: Minimum wavelength for the wavelet transform, defaults to 6.
    :type min_wavelength: int, optional
    :param mult: Multiplicative factor between successive wavelengths, defaults to 2.1.
    :type mult: float, optional
    :param sigma_onf: Ratio of the standard deviation of the Gaussian describing the log Gabor filter's transfer function in the frequency domain to the filter center frequency, defaults to 0.55.
    :type sigma_onf: float, optional
    :param k: Noise compensation factor, defaults to 2.0.
    :type k: float, optional
    :param cut_off: Cut-off value for phase congruency, defaults to 0.5.
    :type cut_off: float, optional
    :param g: Gain factor, defaults to 10.
    :type g: int, optional
    :return: Phase congruency map.
    :rtype: numpy.ndarray
    """
    try:
        logger.info("Starting phase congruency analysis")
        import numpy as np
        import cv2
        import scipy.signal
        import math

        # Compute the mean image over time
        T_mean = np.mean(T, axis=2)
        T_mean = T_mean.astype(np.float32)

        # Initialize parameters
        rows, cols = T_mean.shape
        epsilon = 1e-5  # Small value to avoid division by zero

        # Pre-compute values
        imagefft = np.fft.fft2(T_mean)
        zero = np.zeros((rows, cols))
        total_energy = zero.copy()
        total_sum_an = zero.copy()
        orientation = np.zeros((rows, cols))
        PC = zero.copy()

        # Create Log-Gabor filters
        x, y = np.meshgrid(np.linspace(-0.5, 0.5, cols), np.linspace(-0.5, 0.5, rows))
        radius = np.sqrt(x**2 + y**2)
        radius[rows // 2, cols // 2] = 1  # Avoid division by zero at the center

        theta = np.arctan2(-y, x)
        theta[rows // 2, cols // 2] = 0

        sin_theta = np.sin(theta)
        cos_theta = np.cos(theta)

        # Initialize accumulators
        sum_e__this_orient = zero.copy()
        sum_o_this_orient = zero.copy()
        sum_an_2 = zero.copy()
        energy = zero.copy()

        # Filters for each orientation
        for o in tqdm(range(n_orientation), desc="Processing orientations"):
            angl = o * math.pi / n_orientation
            ds = sin_theta * np.cos(angl) - cos_theta * np.sin(angl)
            dc = cos_theta * np.cos(angl) + sin_theta * np.sin(angl)
            dtheta = np.abs(np.arctan2(ds, dc))
            spread = np.exp((-dtheta**2) / (2 * (math.pi / n_orientation)**2))

            sum_e = zero.copy()
            sum_o = zero.copy()
            sum_an = zero.copy()
            for s in range(n_scale):
                wavelength = min_wavelength * mult**s
                fo = 1.0 / wavelength
                log_gabor = np.exp((-(np.log(radius / fo))**2) / (2 * np.log(sigma_onf)**2))
                log_gabor[radius < (fo / 2)] = 0

                filter_ = log_gabor * spread
                eo = np.fft.ifft2(imagefft * filter_)

                an = np.abs(eo)
                sum_an += an
                sum_e += np.real(eo)
                sum_o += np.imag(eo)

            # Compute phase congruency for this orientation
            energy = np.sqrt(sum_e**2 + sum_o**2) + epsilon
            mean_an = sum_an / n_scale + epsilon
            term = (energy - k * mean_an) / (epsilon + mean_an)
            term = np.maximum(term, 0)
            PC += term

        # Normalize phase congruency
        phase_congruency = PC / n_orientation
        phase_congruency = np.clip(phase_congruency, 0, 1)
        logger.info("Phase congruency analysis completed")

        return phase_congruency

    except Exception as e:
        logger.error(f"Error in phase_congruency_analysis: {e}")
        return None

def dual_tree_cwt_analysis(T, num_levels=4):
    """Perform dual-tree complex wavelet transform analysis on the input thermal signal.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param num_levels: Number of levels for the wavelet transform, defaults to 4.
    :type num_levels: int, optional
    :return: Tuple containing lists of amplitude maps and phase maps for each level and orientation.
    :rtype: Tuple[List[numpy.ndarray], List[numpy.ndarray]]
    """
    try:
        logger.info("Starting dual-tree complex wavelet transform analysis")
        import numpy as np
        import pywt
        from dtcwt import Transform2d

        # Compute the mean image over time to reduce temporal noise
        T_mean = np.mean(T, axis=2)

        # Initialize the Dual-Tree Complex Wavelet Transform
        transform = Transform2d()

        # Perform the forward transform
        coeffs = transform.forward(T_mean, nlevels=num_levels)

        amplitude_maps = []
        phase_maps = []

        # Iterate over levels
        for level in tqdm(range(num_levels), desc="Processing levels"):
            # Get the complex highpass coefficients for this level
            highpass = coeffs.highpasses[level]
            orientation_maps = []
            orientation_phases = []

            # Iterate over orientations (there are 6 orientations in DT-CWT)
            for orientation in range(highpass.shape[2]):
                # Extract the complex coefficients for this orientation
                c = highpass[:, :, orientation]

                # Compute amplitude and phase
                amplitude = np.abs(c)
                phase = np.angle(c)

                amplitude_maps.append(amplitude)
                phase_maps.append(phase)
        logger.info("Dual-tree complex wavelet transform analysis completed")
        return amplitude_maps, phase_maps

    except Exception as e:
        logger.error(f"Error in dual_tree_cwt_analysis: {e}")
        return None, None

def structure_tensor_analysis(T, sigma=1.0):
    """Perform structure tensor analysis on the input thermal signal.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param sigma: Standard deviation for Gaussian kernel used in smoothing, defaults to 1.0.
    :type sigma: float, optional
    :return: Tuple containing coherence and orientation arrays.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)
    """
    try:
        logger.info("Starting structure tensor analysis")
        import numpy as np
        from scipy.ndimage import gaussian_filter, sobel

        # Compute the mean image over time
        T_mean = np.mean(T, axis=2)
        T_mean = T_mean.astype(np.float32)

        # Compute image gradients
        Ix = sobel(T_mean, axis=1)
        Iy = sobel(T_mean, axis=0)

        # Compute products of derivatives at each pixel
        Ixx = gaussian_filter(Ix * Ix, sigma)
        Iyy = gaussian_filter(Iy * Iy, sigma)
        Ixy = gaussian_filter(Ix * Iy, sigma)

        # Compute the eigenvalues of the structure tensor
        lambda1 = (Ixx + Iyy) / 2 + np.sqrt(((Ixx - Iyy) / 2)**2 + Ixy**2)
        lambda2 = (Ixx + Iyy) / 2 - np.sqrt(((Ixx - Iyy) / 2)**2 + Ixy**2)

        # Compute coherence and orientation
        coherence = (lambda1 - lambda2) / (lambda1 + lambda2 + 1e-10)
        orientation = 0.5 * np.arctan2(2 * Ixy, Ixx - Iyy)
        logger.info("Structure tensor analysis completed")

        return coherence, orientation

    except Exception as e:
        logger.error(f"Error in structure_tensor_analysis: {e}")
        return None, None

def phase_stretch_transform(T, warp_strength=0.5, threshold_min=0.1, threshold_max=0.3):
    """Perform phase stretch transform on the input thermal signal.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param warp_strength: Strength of the phase warp, defaults to 0.5.
    :type warp_strength: float, optional
    :param threshold_min: Minimum threshold for phase values, defaults to 0.1.
    :type threshold_min: float, optional
    :param threshold_max: Maximum threshold for phase values, defaults to 0.3.
    :type threshold_max: float, optional
    :return: Binary image after applying phase stretch transform.
    :rtype: numpy.ndarray
    """
    try:
        logger.info("Starting phase stretch transform")
        import numpy as np
        from scipy.fftpack import fft2, ifft2, fftshift

        # Compute the mean image over time
        T_mean = np.mean(T, axis=2)
        T_mean = T_mean.astype(np.float32)

        # Normalize the image
        T_mean = (T_mean - T_mean.min()) / (T_mean.max() - T_mean.min())

        # Perform Fourier Transform
        F = fft2(T_mean)
        F_shifted = fftshift(F)

        # Create frequency coordinates
        rows, cols = T_mean.shape
        u = np.linspace(-0.5, 0.5, cols)
        v = np.linspace(-0.5, 0.5, rows)
        U, V = np.meshgrid(u, v)
        radius = np.sqrt(U**2 + V**2)

        # Apply the phase warp
        phase_kernel = np.exp(-1j * warp_strength * (radius**2))
        F_warped = F_shifted * phase_kernel

        # Inverse Fourier Transform
        F_iwarp = ifft2(fftshift(F_warped))
        phase = np.angle(F_iwarp)

        # Apply thresholding
        pst_output = np.zeros_like(phase)
        mask = (phase > threshold_min) & (phase < threshold_max)
        pst_output[mask] = 1
        logger.info("Phase stretch transform completed")

        return pst_output

    except Exception as e:
        logger.error(f"Error in phase_stretch_transform: {e}")
        return None

def anisotropic_diffusion_filtering(T, num_iterations=10, kappa=50, gamma=0.1, option=1):
    """Perform anisotropic diffusion filtering on the input thermal signal.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param num_iterations: Number of iterations for the diffusion process, defaults to 10.
    :type num_iterations: int, optional
    :param kappa: Conductance coefficient, defaults to 50.
    :type kappa: int, optional
    :param gamma: Integration constant (0 <= gamma <= 0.25 for stability), defaults to 0.1.
    :type gamma: float, optional
    :param option: Option for the diffusion equation (1 for exponential, 2 for reciprocal), defaults to 1.
    :type option: int, optional
    :return: Diffused thermal signal array.
    :rtype: numpy.ndarray
    """
    try:
        logger.info("Starting anisotropic diffusion filtering")
        import numpy as np

        # Compute the mean image over time
        img = np.mean(T, axis=2)
        img = img.astype(np.float32)

        img = img.copy()
        for _ in tqdm(range(num_iterations), desc="Diffusion iterations"):
            # Compute gradients
            deltaN = np.roll(img, -1, axis=0) - img
            deltaS = np.roll(img, 1, axis=0) - img
            deltaE = np.roll(img, -1, axis=1) - img
            deltaW = np.roll(img, 1, axis=1) - img

            # Compute conduction
            if option == 1:
                cN = np.exp(-(deltaN / kappa)**2)
                cS = np.exp(-(deltaS / kappa)**2)
                cE = np.exp(-(deltaE / kappa)**2)
                cW = np.exp(-(deltaW / kappa)**2)
            elif option == 2:
                cN = 1 / (1 + (deltaN / kappa)**2)
                cS = 1 / (1 + (deltaS / kappa)**2)
                cE = 1 / (1 + (deltaE / kappa)**2)
                cW = 1 / (1 + (deltaW / kappa)**2)

            # Update image
            img += gamma * (cN * deltaN + cS * deltaS + cE * deltaE + cW * deltaW)

        diffused_image = img
        logger.info("Anisotropic diffusion filtering completed")
        return diffused_image

    except Exception as e:
        logger.error(f"Error in anisotropic_diffusion_filtering: {e}")
        return None

def entropy_based_imaging(T, window_size=9):
    """Perform entropy-based imaging on the input thermal signal.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param window_size: Size of the window used for entropy calculation, defaults to 9.
    :type window_size: int, optional
    :return: Entropy map of the input thermal signal.
    :rtype: numpy.ndarray
    """
    try:
        logger.info("Starting entropy-based imaging")
        import numpy as np
        from skimage.util import view_as_windows
        from scipy.stats import entropy

        # Compute the mean image over time
        img = np.mean(T, axis=2)
        img = img.astype(np.float32)

        # Normalize the image
        img = (img - img.min()) / (img.max() - img.min())

        # Pad the image to handle borders
        pad_size = window_size // 2
        img_padded = np.pad(img, pad_size, mode='reflect')

        # Create an empty entropy map
        entropy_map = np.zeros_like(img)

        # Compute entropy for each local window
        for i in tqdm(range(entropy_map.shape[0]), desc="Computing entropy"):
            for j in range(entropy_map.shape[1]):
                window = img_padded[i:i+window_size, j:j+window_size]
                hist, _ = np.histogram(window, bins=256, range=(0, 1), density=True)
                entropy_map[i, j] = entropy(hist + 1e-10)
        logger.info("Entropy-based imaging completed")
        return entropy_map

    except Exception as e:
        logger.error(f"Error in entropy_based_imaging: {e}")
        return None

def dtw_clustering_defect_detection(T, n_clusters=4, downsample_factor=1, max_iter=10, n_jobs=-1):
    """
    Perform DTW clustering for defect detection on the input thermal signal.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param n_clusters: Number of clusters for the DTW clustering, defaults to 4.
    :type n_clusters: int, optional
    :param downsample_factor: Factor by which to downsample the frames, defaults to 1 (no downsampling).
    :type downsample_factor: int, optional
    :param max_iter: Maximum number of iterations for the clustering algorithm, defaults to 10.
    :type max_iter: int, optional
    :param n_jobs: Number of parallel jobs to run, defaults to 1.
    :type n_jobs: int, optional
    :return: Defect map indicating cluster labels for each pixel.
    :rtype: numpy.ndarray
    """
    try:
        from tslearn.clustering import TimeSeriesKMeans
        from tslearn.metrics import cdist_dtw
        logger.info("Starting DTW clustering for defect detection")
        
        # Downsample the frames if needed
        if downsample_factor > 1:
            T = T[:, :, ::downsample_factor]
            logger.info(f"Downsampled frames by a factor of {downsample_factor}")
        
        height, width, frames = T.shape
        T_reshaped = T.reshape(-1, frames).astype(np.float32)  # Use float32 to save memory

        # Optional: Normalize the time series to have zero mean and unit variance
        T_mean = T_reshaped.mean(axis=1, keepdims=True)
        T_std = T_reshaped.std(axis=1, keepdims=True) + 1e-8  # Avoid division by zero
        T_normalized = (T_reshaped - T_mean) / T_std

        logger.info(f"Data reshaped to {T_normalized.shape}")

        # Perform clustering with limited parallelism
        km_dtw = TimeSeriesKMeans(
            n_clusters=n_clusters, 
            metric="dtw", 
            max_iter=max_iter, 
            n_jobs=n_jobs,  # Limit the number of parallel jobs
            verbose=True,    # Enable verbosity for progress tracking
            random_state=42  # For reproducibility
        )
        cluster_labels = km_dtw.fit_predict(T_normalized)

        # Reshape cluster labels into image
        defect_map = cluster_labels.reshape(height, width)
        logger.info("DTW clustering completed successfully")
        return defect_map
    except MemoryError:
        logger.error("MemoryError: The operation ran out of memory. Consider reducing data size or parameters.")
        return None
    except Exception as e:
        logger.error(f"Error in dtw_clustering_defect_detection: {e}")
        return None

def frequency_ratio_imaging(T, fs, f_stim):
    """Perform frequency ratio imaging on the input thermal signal.

    :param T: Input thermal signal array with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param fs: Sampling frequency of the input signal.
    :type fs: float
    :param f_stim: Stimulation frequency for the imaging techniques.
    :type f_stim: float
    :return: Defect map indicating the ratio of harmonic to fundamental frequency amplitude for each pixel.
    :rtype: numpy.ndarray
    """
    try:
        logger.info("Starting frequency ratio imaging")
        height, width, frames = T.shape
        defect_map = np.zeros((height, width))

        freqs = np.fft.fftfreq(frames, d=1/fs)
        idx_fundamental = np.argmin(np.abs(freqs - f_stim))
        idx_harmonic = np.argmin(np.abs(freqs - 2*f_stim))

        for i in tqdm(range(height), desc="Computing frequency ratios"):
            for j in range(width):
                # FFT of the signal
                signal_fft = np.fft.fft(T[i, j, :])
                # Amplitude at fundamental and harmonic frequencies
                amp_fundamental = np.abs(signal_fft[idx_fundamental])
                amp_harmonic = np.abs(signal_fft[idx_harmonic])
                # Compute ratio
                if amp_fundamental != 0:
                    defect_map[i, j] = amp_harmonic / amp_fundamental
                else:
                    defect_map[i, j] = 0

        # Normalize defect map
        defect_map = (defect_map - np.min(defect_map)) / (np.max(defect_map) - np.min(defect_map))
        logger.info("Frequency ratio imaging completed")
        return defect_map
    except Exception as e:
        logger.error(f"Error in frequency_ratio_imaging: {e}")
        return None

def coherence_map(frame):
    """Compute the coherence map of the input frame.

    :param frame: Input 2D array representing a single frame.
    :type frame: numpy.ndarray
    :return: Coherence map of the input frame.
    :rtype: numpy.ndarray
    """
    try:
        logger.info("Starting coherence map computation")
        height, width = frame.shape
        phase_diff = np.zeros((height, width), dtype=np.float16)

        for i in tqdm(range(1, height - 1), desc="Computing coherence map"):
            for j in range(1, width - 1):
                neighbors = [
                    frame[i - 1, j],
                    frame[i + 1, j],
                    frame[i, j - 1],
                    frame[i, j + 1],
                    frame[i - 1, j - 1],
                    frame[i - 1, j + 1],
                    frame[i + 1, j - 1],
                    frame[i + 1, j + 1]
                ]
                phase_diff[i, j] = np.std([frame[i, j] - neighbor for neighbor in neighbors])

        # Avoid division by zero
        max_diff = np.max(phase_diff)
        if max_diff == 0:
            coherence = np.ones((height, width), dtype=np.float16)
        else:
            coherence = 1 - (phase_diff / max_diff)

        logger.info("Coherence map computation completed")
        return coherence

    except Exception as e:
        logger.error(f"Error in coherence_map: {e}")
        return None

