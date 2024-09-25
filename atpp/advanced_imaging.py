import numpy as np
from scipy.signal import hilbert
from sklearn.decomposition import PCA
import pywt
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import os
from multiprocessing import cpu_count


# Retrieve number of processors automatically
NUM_PROCESSORS = os.cpu_count() or cpu_count()  # Fallback to multiprocessing if os.cpu_count() fails


#!!!!!!!!!!!!!!!! ATTENZIONE A PARALLELIZZARE CP.SUM E =+  ||||||||||||||||||||



# Try importing cupy for GPU-accelerated computing
try:
    import cupy as cp
    USE_GPU = True
    #free memory
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()    
    print("Using GPU-accelerated computing")
except ImportError:
    USE_GPU = False
    cp = np  # Fallback to numpy if GPU is not available
    print("Using CPU-based computing")


def clear_gpu_memory():
    """ Clear GPU memory by deleting arrays and synchronizing. """
    cp.get_default_memory_pool().free_all_blocks()
    cp.cuda.Stream.null.synchronize()

def get_max_chunk_frames(height, width, dtype, extra_arrays=2, overhead=0.9):
    """
    Calculate the maximum number of frames that can be processed at once without exceeding GPU memory.

    Parameters:
    - height: int
    - width: int
    - dtype: data type (e.g., cp.float16)
    - extra_arrays: int, number of extra arrays of the same size as T_gpu required during processing
    - overhead: float, fraction of the free memory to use (e.g., 0.9 means use 90% of free memory)

    Returns:
    - max_frames: int, maximum number of frames that can be processed at once
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
    """
    Calculate the maximum number of pixels that can be processed at once without exceeding GPU memory.

    Parameters:
    - frames: int
    - dtype: data type (e.g., cp.float16)
    - extra_arrays: int, number of extra arrays of the same size as T_gpu required during processing
    - overhead: float, fraction of the free memory to use (e.g., 0.9 means use 90% of free memory)

    Returns:
    - max_pixels: int, maximum number of pixels that can be processed at once
    """
    if USE_GPU:
        # Synchronize and clear the CuPy memory pool
        cp.cuda.Device().synchronize()
        cp.get_default_memory_pool().free_all_blocks()

        # Get the correct memory information
        free_mem, total_mem = cp.cuda.Device().mem_info
        # print(f"Total Memory: {total_mem / (1024**2)} MB, Free Memory: {free_mem / (1024**2)} MB")

        free_mem *= overhead  # Use only a fraction of the free memory

        bytes_per_element = cp.dtype(dtype).itemsize
        per_pixel_memory = frames * bytes_per_element * (1 + extra_arrays)
        
        # print(f"Free memory: {free_mem} bytes")
        # print(f"Per pixel memory: {per_pixel_memory} bytes")
        # print(f"Bytes per element: {bytes_per_element}")

        max_pixels = int(free_mem / per_pixel_memory)
        max_pixels = max(1, max_pixels)  # Ensure at least one pixel is processed
        return max_pixels
    else:
        return None  # Not applicable for CPU processing


def phase_coherence_imaging(T, fs, f_stim):
    """
    Perform phase coherence imaging on a 3D array of time-domain signals to extract amplitude, phase, and phase coherence maps.

    Parameters
    ----------
    T : numpy.ndarray or cupy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    fs : float
        Sampling frequency of the signal (samples per second).
    f_stim : float
        Stimulation frequency used for phase coherence imaging (Hz).

    Returns
    -------
    tuple
        A tuple containing amplitude, phase, and phase coherence maps as 2D numpy arrays.
    """
    try:
        if USE_GPU:
            # Convert T to CuPy array for GPU processing
            T_gpu = cp.asarray(T, dtype=cp.float16)
            height, width, frames = T_gpu.shape

            # Create time vector
            t = cp.arange(frames) / fs

            # Reference signals for demodulation
            ref_sin = cp.sin(2 * cp.pi * f_stim * t)
            ref_cos = cp.cos(2 * cp.pi * f_stim * t)

            # Reshape reference signals for broadcasting
            ref_sin = ref_sin[cp.newaxis, cp.newaxis, :]
            ref_cos = ref_cos[cp.newaxis, cp.newaxis, :]

            # Compute I and Q using vectorized operations on GPU
            I_gpu = cp.sum(T_gpu * ref_cos, axis=2) / frames
            Q_gpu = cp.sum(T_gpu * ref_sin, axis=2) / frames

            # Transfer results back to CPU
            I = I_gpu.get()
            Q = Q_gpu.get()

            # Compute amplitude and phase
            amplitude = np.sqrt(I**2 + Q**2)
            phase = np.arctan2(Q, I)

            # Phase coherence computation
            phase_diff = np.zeros((height, width), dtype=np.float16)
            for i in range(1, height - 1):
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
            if max_diff == 0:
                phase_coherence = np.ones((height, width), dtype=np.float16)
            else:
                phase_coherence = 1 - (phase_diff / max_diff)

            # Clear GPU memory
            clear_gpu_memory()

        else:
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

                Parameters
                ----------
                i : int
                    Row index.

                Returns
                -------
                tuple
                    Tuple containing row index, I_row, and Q_row.
                """
                row_I = np.sum(T[i, :, :] * ref_cos, axis=1) / frames
                row_Q = np.sum(T[i, :, :] * ref_sin, axis=1) / frames
                return i, row_I, row_Q

            # Use ProcessPoolExecutor to parallelize row computations
            with ProcessPoolExecutor(max_workers=NUM_PROCESSORS) as executor:
                for i, row_I, row_Q in executor.map(calculate_iq, range(height)):
                    I[i, :] = row_I
                    Q[i, :] = row_Q

            # Compute amplitude and phase
            amplitude = np.sqrt(I**2 + Q**2)
            phase = np.arctan2(Q, I)

            # Phase coherence computation
            phase_diff = np.zeros((height, width), dtype=np.float16)
            for i in range(1, height - 1):
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

        return amplitude, phase, phase_coherence

    except Exception as e:
        print(f"Error in phase_coherence_imaging: {e}")
        return None, None, None


def synchronous_demodulation(T, fs, f_stim):
    """
    Perform synchronous demodulation on a 3D array of time-domain signals to extract amplitude and phase maps.

    Parameters
    ----------
    T : numpy.ndarray or cupy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    fs : float
        Sampling frequency of the signal (samples per second).
    f_stim : float
        Stimulation frequency used for demodulation (Hz).

    Returns
    -------
    tuple
        A tuple containing amplitude and phase maps as 2D numpy arrays.
    """
    try:
        if USE_GPU:
            # Convert T to CuPy array for GPU processing
            T_gpu = cp.asarray(T, dtype=cp.float16)
            height, width, frames = T_gpu.shape

            # Create time vector
            t = cp.arange(frames) / fs

            # Reference signals for demodulation
            ref_sin = cp.sin(2 * cp.pi * f_stim * t)
            ref_cos = cp.cos(2 * cp.pi * f_stim * t)

            # Reshape reference signals for broadcasting
            ref_sin = ref_sin[cp.newaxis, cp.newaxis, :]
            ref_cos = ref_cos[cp.newaxis, cp.newaxis, :]

            # Compute I and Q using vectorized operations on GPU
            I_gpu = cp.sum(T_gpu * ref_cos, axis=2) / frames
            Q_gpu = cp.sum(T_gpu * ref_sin, axis=2) / frames

            # Transfer results back to CPU
            I = I_gpu.get()
            Q = Q_gpu.get()

            # Compute amplitude and phase
            amplitude = np.sqrt(I**2 + Q**2)
            phase = np.arctan2(Q, I)

            # Clear GPU memory
            clear_gpu_memory()

        else:
            # CPU-based processing
            height, width, frames = T.shape

            # Create time vector
            t = np.arange(frames) / fs

            # Reference signals for demodulation
            ref_sin = np.sin(2 * np.pi * f_stim * t)
            ref_cos = np.cos(2 * np.pi * f_stim * t)

            # Reshape reference signals for broadcasting
            ref_sin = ref_sin[np.newaxis, np.newaxis, :]
            ref_cos = ref_cos[np.newaxis, np.newaxis, :]

            # Initialize I and Q arrays
            I = np.zeros((height, width), dtype=np.float16)
            Q = np.zeros((height, width), dtype=np.float16)

            def calculate_iq(i):
                """
                Calculate I and Q for a specific row.

                Parameters
                ----------
                i : int
                    Row index.

                Returns
                -------
                tuple
                    Tuple containing row index, I_row, and Q_row.
                """
                row_I = np.sum(T[i, :, :] * ref_cos, axis=1) / frames
                row_Q = np.sum(T[i, :, :] * ref_sin, axis=1) / frames
                return i, row_I, row_Q

            # Use ProcessPoolExecutor to parallelize row computations
            with ProcessPoolExecutor(max_workers=NUM_PROCESSORS) as executor:
                for i, row_I, row_Q in executor.map(calculate_iq, range(height)):
                    I[i, :] = row_I
                    Q[i, :] = row_Q

            # Compute amplitude and phase
            amplitude = np.sqrt(I**2 + Q**2)
            phase = np.arctan2(Q, I)

        return amplitude, phase

    except Exception as e:
        print(f"Error in synchronous_demodulation: {e}")
        return None, None


def hilbert_transform_analysis(T):
    """
    Perform Hilbert transform analysis on a 3D array of time-domain signals to extract amplitude and phase maps.

    Parameters
    ----------
    T : numpy.ndarray or cupy.ndarray
        3D array of thermal data with dimensions (height, width, frames).

    Returns
    -------
    tuple
        A tuple containing amplitude and phase maps as 2D numpy arrays.
    """
    try:
        if USE_GPU:
            height, width, frames = T.shape

            # Calculate the maximum number of pixels we can process at once
            max_chunk_pixels = get_max_chunk_pixels(frames, cp.complex128, extra_arrays=7, overhead=0.9)
            total_pixels = height * width
            print(f"max_chunk_pixels: {max_chunk_pixels}, total_pixels: {total_pixels}")

            # Initialize amplitude and phase arrays
            amplitude = np.zeros((height, width, frames), dtype=np.float16)
            phase = np.zeros((height, width, frames), dtype=np.float16)

            rows_per_chunk = max_chunk_pixels // width
            row_chunks = (height) // rows_per_chunk

            if row_chunks != 1 and row_chunks != 0:
                print(f"Needed chunks: {row_chunks}")
            elif row_chunks == 0:
                row_chunks = 1
                print(f"Correcting --> Needed chunks: {row_chunks}")

            for chunk_idx in range(row_chunks):
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

        else:
            # CPU-based processing remains unchanged
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
                for i, row_amp, row_ph in executor.map(calculate_hilbert, range(height)):
                    amplitude[i, :] = row_amp
                    phase[i, :] = row_ph

        return amplitude, phase

    except Exception as e:
        print(f"Error in hilbert_transform_analysis: {e}")
        return None, None


def thermal_signal_reconstruction(T, order=5):
    """ Reconstruct thermal signals using polynomial fitting. STILL TO BE TESTED """
    height, width, frames = T.shape
    log_time = np.log(np.arange(1, frames + 1))
    T_reconstructed = np.zeros_like(T)

    def reconstruct_signal(i):
        for j in range(width):
            signal = T[i, j, :]
            log_signal = np.log(signal + np.finfo(float).eps)
            coeffs = np.polyfit(log_time, log_signal, order)
            log_signal_fit = np.polyval(coeffs, log_time)
            T_reconstructed[i, j, :] = np.exp(log_signal_fit)

    with ProcessPoolExecutor(max_workers=NUM_PROCESSORS) as executor:
        executor.map(reconstruct_signal, range(height))

    return T_reconstructed


def modulated_thermography(T, fs, f_stim, harmonics=[2, 3]):
    """
    Perform modulated thermography analysis on a 3D array of time-domain signals.

    Parameters
    ----------
    T : numpy.ndarray
        3D array of time-domain signals with dimensions (height, width, frames).
    fs : float
        Sampling frequency of the signal (samples per second).
    f_stim : float
        Frequency of the stimulus signal (Hz).
    harmonics : list, optional
        List of harmonics to analyze (default is [2, 3]).

    Returns
    -------
    tuple
        A tuple containing dictionaries of amplitude and phase images for each harmonic.

    Example
    -------
    >>> T = np.random.rand(100, 100, 1000)
    >>> fs = 1000.0
    >>> f_stim = 10.0
    >>> amplitude, phase = modulated_thermography(T, fs, f_stim, harmonics=[2, 3])
    """
    # Convert T to cupy array if GPU is used
    T_gpu = cp.asarray(T) if USE_GPU else T

    height, width, frames = T_gpu.shape
    amplitude = {}
    phase = {}

    def demodulate_harmonic(h):
        # Ensure `np.arange(frames)` is compatible with cupy by converting to cupy array if needed
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
                executor.map(calculate_iq, range(height))
        else:
            for i in range(height):
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
    return amplitude, phase


def principal_component_thermography(T, n_components=5):
    """ Perform Principal Component Thermography (PCT) on a 3D array of time-domain signals. """
    height, width, frames = T.shape
    data = T.reshape(-1, frames)
    data_mean = np.mean(data, axis=0)
    data_centered = data - data_mean
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(data_centered)
    pcs_images = [pc.reshape(height, width) for pc in principal_components.T]
    return pcs_images


def pulsed_phase_thermography(T, fs):
    """
    Perform Pulsed Phase Thermography (PPT) on a 3D array of time-domain signals.

    Parameters
    ----------
    T : numpy.ndarray or cupy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    fs : float
        Sampling frequency of the signal (samples per second).


    Returns
    -------
    tuple
        A tuple containing amplitude and phase images at different frequencies, and the frequency values.

    Example
    -------
    >>> T = np.random.rand(100, 100, 1000)
    >>> fs = 1000.0
    >>> amplitude, phase, freqs = pulsed_phase_thermography(T, fs)
    """
    try:
        if USE_GPU:
            # Convert T to CuPy array for GPU processing
            T_gpu = cp.asarray(T, dtype=cp.float16)
            height, width, frames = T_gpu.shape
            print(f"Processing on GPU with shape: {T_gpu.shape}")

            # Initialize list to collect FFT batches
            fft_batches = []
            
            chunks= get_max_chunk_frames(height, width, cp.complex128, extra_arrays=7, overhead=0.9)

            # Process data in batches to manage GPU memory
            for start in range(0, frames, chunks):
                end = min(start + chunks, frames)
                print(f"Processing frames {start} to {end} on GPU")

                # Perform FFT on the current batch
                batch_fft = cp.fft.fft(T_gpu[:, :, start:end], axis=2)

                # Append the FFT result to the list
                fft_batches.append(batch_fft)

                # Optionally synchronize and free memory after each batch
                cp.cuda.Stream.null.synchronize()
                cp.get_default_memory_pool().free_all_blocks()

            # Concatenate all FFT batches along the frame axis
            fft_data = cp.concatenate(fft_batches, axis=2)
            print(f"Concatenated FFT data shape: {fft_data.shape}")

            # Clear the list to free memory
            del fft_batches
            cp.get_default_memory_pool().free_all_blocks()

            # Get the frequencies corresponding to the FFT result
            freqs = cp.fft.fftfreq(frames, d=1/fs)
            pos_mask = freqs > 0

            # Apply the positive frequency mask
            fft_data = fft_data[:, :, pos_mask]
            freqs = freqs[pos_mask]
            print(f"Positive frequencies count: {cp.sum(pos_mask)}")

            # Calculate amplitude and phase
            amplitude = cp.abs(fft_data)
            phase = cp.angle(fft_data)

            # Transfer results back to NumPy arrays
            amplitude = amplitude.get()
            phase = phase.get()
            freqs = freqs.get()

            # Clear GPU memory
            clear_gpu_memory()

        else:
            # Perform FFT on CPU using NumPy
            height, width, frames = T.shape
            print(f"Processing on CPU with shape: {T.shape}")

            # Perform FFT along the time axis
            fft_data = np.fft.fft(T, axis=2)
            print(f"FFT data shape: {fft_data.shape}")

            # Get the frequencies corresponding to the FFT result
            freqs = np.fft.fftfreq(frames, d=1/fs)
            pos_mask = freqs > 0
            print(f"Positive frequencies count: {np.sum(pos_mask)}")

            # Apply the positive frequency mask
            fft_data = fft_data[:, :, pos_mask]
            freqs = freqs[pos_mask]

            # Calculate amplitude and phase
            amplitude = np.abs(fft_data)
            phase = np.angle(fft_data)

        return amplitude, phase, freqs

    except cp.cuda.memory.OutOfMemoryError as e:
        print(f"GPU OutOfMemoryError: {e}")
        clear_gpu_memory()
        raise e
    except Exception as e:
        print(f"Error in pulsed_phase_thermography: {e}")
        return None, None, None


def wavelet_transform_analysis(T, wavelet='db4', level=3):
    """ Perform Wavelet Transform Analysis on a 3D array of thermal data. """
    height, width, frames = T.shape
    coeffs = []

    def calculate_wavelet(i):
        for j in range(width):
            signal = T[i, j, :]
            coeff = pywt.wavedec(signal, wavelet, level=level)
            coeffs.append(coeff)

    with ProcessPoolExecutor(max_workers=NUM_PROCESSORS) as executor:
        executor.map(calculate_wavelet, range(height))

    return coeffs


def visualize_comparison(T, fs, f_stim, time):
    """
    Visualize and compare the results of various imaging models.

    :param T: 3D array of temperature data (height x width x frames).
    :param fs: Sampling frequency.
    :param f_stim: Stimulus frequency for phase coherence or modulation analysis.
    :param time: Time vector corresponding to frames.
    """
    models = {
        "Phase Coherence Imaging": phase_coherence_imaging(T, fs, f_stim),
        "Synchronous Demodulation": synchronous_demodulation(T, fs, f_stim),
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


def visualize_wavelet_coefficients(T, wavelet='db4', level=3):
    """
    STILL to be refactored
    Visualizes wavelet coefficients of a 3D thermal data array using scalograms.

    Parameters
    ----------
    T : numpy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    wavelet : str, optional
        The type of wavelet to use (default is 'db4').
    level : int, optional
        The level of decomposition (default is 3).

    Returns
    -------
    None
    """
    # Get dimensions
    height, width, frames = T.shape
    coeffs_list = []

    # Apply wavelet transform to each pixel time series
    for i in range(height):
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



    """
    Perform Independent Component Thermography (ICT) on a 3D array of time-domain signals.

    Parameters
    ----------
    T : numpy.ndarray
        3D array of time-domain signals with dimensions (height, width, frames).
    n_components : int, optional
        Number of independent components to extract (default is 5).

    Returns
    -------
    ict_images : list of numpy.ndarray
        List of independent component images.

    Example
    -------
    >>> T = np.random.rand(100, 100, 1000)
    >>> ict_images = independent_component_thermography(T, n_components=5)
    """
    
    try:
        # Get dimensions
        height, width, frames = T.shape

        # Reshape T to (pixels, frames)
        data = T.reshape(-1, frames)
        data_mean = np.mean(data, axis=0)
        data_centered = data - data_mean

        # Perform ICA
        from sklearn.decomposition import FastICA        
        from skimage.transform import resize

        ica = FastICA(n_components=n_components, random_state=0)
        independent_components = ica.fit_transform(data_centered.T).T
        #reconstruct images with inverse transform
        reconstructed_data = ica.inverse_transform(independent_components.T).T


        return reconstructed_data.reshape(height, width, frames)

    except Exception as e:
        print(f"Error in independent_component_thermography: {e}")
        return None




def monogenic_signal_analysis(T):
    """
    Perform Monogenic Signal Analysis on a 3D array of thermal data to extract local amplitude and phase maps.
    
    Parameters
    ----------
    T : numpy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    
    Returns
    -------
    amplitude : numpy.ndarray
        Local amplitude map.
    phase : numpy.ndarray
        Local phase map.
    orientation : numpy.ndarray
        Local orientation map.
    
    Example
    -------
    >>> amplitude, phase, orientation = monogenic_signal_analysis(T)
    """
    try:
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

        return amplitude, phase, orientation

    except Exception as e:
        print(f"Error in monogenic_signal_analysis: {e}")
        return None, None, None

# def monogenic_signal_analysis_gpu(T):
#     """
#     GPU-accelerated version of Monogenic Signal Analysis.
#     """
#     try:
#         import cupy as cp
#         from cupyx.scipy.fft import fftn, ifftn
#         height, width, frames = T.shape

#         # Convert T to GPU array
#         T_gpu = cp.asarray(T, dtype=cp.float32)

#         # Compute the mean image over time
#         T_mean_gpu = cp.mean(T_gpu, axis=2)

#         # Compute the Riesz transform kernels
#         u = cp.fft.fftfreq(height).reshape(-1, 1)
#         v = cp.fft.fftfreq(width).reshape(1, -1)
#         radius = cp.sqrt(u**2 + v**2) + cp.finfo(cp.float32).eps  # Avoid division by zero

#         # Riesz kernels
#         R1 = -1j * u / radius
#         R2 = -1j * v / radius

#         # Perform Fourier transform of the mean image
#         F = fftn(T_mean_gpu)

#         # Apply Riesz transform
#         R1F = R1 * F
#         R2F = R2 * F

#         # Inverse Fourier transform to get spatial domain representations
#         monogenic_R1 = cp.real(ifftn(R1F))
#         monogenic_R2 = cp.real(ifftn(R2F))

#         # Compute local amplitude
#         amplitude_gpu = cp.sqrt(T_mean_gpu**2 + monogenic_R1**2 + monogenic_R2**2)

#         # Compute local phase
#         phase_gpu = cp.arctan2(cp.sqrt(monogenic_R1**2 + monogenic_R2**2), T_mean_gpu)

#         # Compute local orientation
#         orientation_gpu = cp.arctan2(monogenic_R2, monogenic_R1)

#         # Transfer results back to CPU
#         amplitude = cp.asnumpy(amplitude_gpu)
#         phase = cp.asnumpy(phase_gpu)
#         orientation = cp.asnumpy(orientation_gpu)

#         # Clear GPU memory
#         cp.get_default_memory_pool().free_all_blocks()

#         return amplitude, phase, orientation

#     except Exception as e:
#         print(f"Error in monogenic_signal_analysis_gpu: {e}")
#         return None, None, None


def phase_congruency_analysis(T, n_scale=4, n_orientation=4, min_wavelength=6, mult=2.1, sigma_onf=0.55, k=2.0, cut_off=0.5, g=10):
    """
    Perform Phase Congruency Analysis on a 3D array of thermal data to extract phase congruency maps.
    
    Parameters
    ----------
    T : numpy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    n_scale : int, optional
        Number of wavelet scales (default is 4).
    n_orientation : int, optional
        Number of filter orientations (default is 4).
    min_wavelength : int, optional
        Wavelength of the smallest scale filter (default is 6).
    mult : float, optional
        Scaling factor between successive filters (default is 2.1).
    sigma_onf : float, optional
        Ratio of the standard deviation of the Gaussian describing the log Gabor filter's transfer function (default is 0.55).
    k : float, optional
        Noisy input parameter (default is 2.0).
    cut_off : float, optional
        Fractional measure of frequency spread below which phase congruency values get penalized (default is 0.5).
    g : float, optional
        Controls the sharpness of the transition in the sigmoid function used to weight phase congruency for frequency spread (default is 10).
    
    Returns
    -------
    phase_congruency : numpy.ndarray
        Phase congruency map.
    
    Example
    -------
    >>> phase_congruency = phase_congruency_analysis(T)
    """
    try:
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
        for o in range(n_orientation):
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

        return phase_congruency

    except Exception as e:
        print(f"Error in phase_congruency_analysis: {e}")
        return None

# def phase_congruency_analysis_gpu(T, n_scale=4, n_orientation=4, min_wavelength=6, mult=2.1, sigma_onf=0.55, k=2.0, cut_off=0.5, g=10):
#     """
#     GPU-accelerated version of Phase Congruency Analysis.
#     """
#     try:
#         import cupy as cp

#         # Compute the mean image over time
#         T_gpu = cp.asarray(T, dtype=cp.float32)
#         T_mean_gpu = cp.mean(T_gpu, axis=2)
#         rows, cols = T_mean_gpu.shape
#         epsilon = 1e-5  # Small value to avoid division by zero

#         # Pre-compute values
#         imagefft = cp.fft.fft2(T_mean_gpu)
#         zero = cp.zeros((rows, cols), dtype=cp.float32)
#         PC = zero.copy()

#         # Create Log-Gabor filters
#         x, y = cp.meshgrid(cp.linspace(-0.5, 0.5, cols), cp.linspace(-0.5, 0.5, rows))
#         radius = cp.sqrt(x**2 + y**2)
#         radius[rows // 2, cols // 2] = 1  # Avoid division by zero at the center

#         theta = cp.arctan2(-y, x)
#         theta[rows // 2, cols // 2] = 0

#         sin_theta = cp.sin(theta)
#         cos_theta = cp.cos(theta)

#         # Filters for each orientation
#         for o in range(n_orientation):
#             angl = o * cp.pi / n_orientation
#             ds = sin_theta * cp.cos(angl) - cos_theta * cp.sin(angl)
#             dc = cos_theta * cp.cos(angl) + sin_theta * cp.sin(angl)
#             dtheta = cp.abs(cp.arctan2(ds, dc))
#             spread = cp.exp((-dtheta**2) / (2 * (cp.pi / n_orientation)**2))

#             sum_e = zero.copy()
#             sum_o = zero.copy()
#             sum_an = zero.copy()
#             for s in range(n_scale):
#                 wavelength = min_wavelength * mult**s
#                 fo = 1.0 / wavelength
#                 log_gabor = cp.exp((-(cp.log(radius / fo))**2) / (2 * cp.log(sigma_onf)**2))
#                 log_gabor[radius < (fo / 2)] = 0

#                 filter_ = log_gabor * spread
#                 eo = cp.fft.ifft2(imagefft * filter_)

#                 an = cp.abs(eo)
#                 sum_an += an
#                 sum_e += cp.real(eo)
#                 sum_o += cp.imag(eo)

#             # Compute phase congruency for this orientation
#             energy = cp.sqrt(sum_e**2 + sum_o**2) + epsilon
#             mean_an = sum_an / n_scale + epsilon
#             term = (energy - k * mean_an) / (epsilon + mean_an)
#             term = cp.maximum(term, 0)
#             PC += term

#         # Normalize phase congruency
#         phase_congruency_gpu = PC / n_orientation
#         phase_congruency_gpu = cp.clip(phase_congruency_gpu, 0, 1)

#         # Transfer results back to CPU
#         phase_congruency = cp.asnumpy(phase_congruency_gpu)

#         # Clear GPU memory
#         cp.get_default_memory_pool().free_all_blocks()

#         return phase_congruency

#     except Exception as e:
#         print(f"Error in phase_congruency_analysis_gpu: {e}")
#         return None

def dual_tree_cwt_analysis(T, num_levels=4):
    """
    Perform Dual-Tree Complex Wavelet Transform Analysis on a 3D array of thermal data
    to extract amplitude and phase maps.

    Parameters
    ----------
    T : numpy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    num_levels : int, optional
        Number of decomposition levels (default is 4).

    Returns
    -------
    amplitude_maps : list of numpy.ndarray
        List containing amplitude maps at each level and orientation.
    phase_maps : list of numpy.ndarray
        List containing phase maps at each level and orientation.

    Example
    -------
    >>> amplitude_maps, phase_maps = dual_tree_cwt_analysis(T)
    """
    try:
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
        for level in range(num_levels):
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

        return amplitude_maps, phase_maps

    except Exception as e:
        print(f"Error in dual_tree_cwt_analysis: {e}")
        return None, None


def structure_tensor_analysis(T, sigma=1.0):
    """
    Perform Structure Tensor Analysis on a 3D array of thermal data to extract coherence and orientation maps.
    
    Parameters
    ----------
    T : numpy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    sigma : float, optional
        Standard deviation for Gaussian kernel used in smoothing (default is 1.0).
    
    Returns
    -------
    coherence : numpy.ndarray
        Coherence map indicating the degree of local anisotropy.
    orientation : numpy.ndarray
        Orientation map indicating the local dominant orientation.
    
    Example
    -------
    >>> coherence, orientation = structure_tensor_analysis(T)
    """
    try:
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

        return coherence, orientation

    except Exception as e:
        print(f"Error in structure_tensor_analysis: {e}")
        return None, None


def phase_stretch_transform(T, warp_strength=0.5, threshold_min=0.1, threshold_max=0.3):
    """
    Perform Phase Stretch Transform on a 3D array of thermal data to extract feature-enhanced maps.
    
    Parameters
    ----------
    T : numpy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    warp_strength : float, optional
        Strength of the phase warp (default is 0.5).
    threshold_min : float, optional
        Minimum threshold for edge detection (default is 0.1).
    threshold_max : float, optional
        Maximum threshold for edge detection (default is 0.3).
    
    Returns
    -------
    pst_output : numpy.ndarray
        Feature-enhanced map after applying the Phase Stretch Transform.
    
    Example
    -------
    >>> pst_output = phase_stretch_transform(T)
    """
    try:
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

        return pst_output

    except Exception as e:
        print(f"Error in phase_stretch_transform: {e}")
        return None


def anisotropic_diffusion_filtering(T, num_iterations=10, kappa=50, gamma=0.1, option=1):
    """
    Perform Anisotropic Diffusion Filtering on a 3D array of thermal data to enhance edges.
    
    Parameters
    ----------
    T : numpy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    num_iterations : int, optional
        Number of iterations to run the diffusion process (default is 10).
    kappa : float, optional
        Conduction coefficient (default is 50).
    gamma : float, optional
        Integration constant (default is 0.1).
    option : int, optional
        Conductivity function option: 1 or 2 (default is 1).
    
    Returns
    -------
    diffused_image : numpy.ndarray
        Edge-enhanced image after anisotropic diffusion.
    
    Example
    -------
    >>> diffused_image = anisotropic_diffusion_filtering(T)
    """
    try:
        import numpy as np

        # Compute the mean image over time
        img = np.mean(T, axis=2)
        img = img.astype(np.float32)

        img = img.copy()
        for _ in range(num_iterations):
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
        return diffused_image

    except Exception as e:
        print(f"Error in anisotropic_diffusion_filtering: {e}")
        return None


def entropy_based_imaging(T, window_size=9):
    """
    Perform Entropy-Based Imaging on a 3D array of thermal data to extract entropy maps.
    
    Parameters
    ----------
    T : numpy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    window_size : int, optional
        Size of the local window to compute entropy (default is 9).
    
    Returns
    -------
    entropy_map : numpy.ndarray
        Local entropy map of the image.
    
    Example
    -------
    >>> entropy_map = entropy_based_imaging(T)
    """
    try:
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
        for i in range(entropy_map.shape[0]):
            for j in range(entropy_map.shape[1]):
                window = img_padded[i:i+window_size, j:j+window_size]
                hist, _ = np.histogram(window, bins=256, range=(0, 1), density=True)
                entropy_map[i, j] = entropy(hist + 1e-10)

        return entropy_map

    except Exception as e:
        print(f"Error in entropy_based_imaging: {e}")
        return None


# def synchrosqueezed_wavelet_transform(T):
#     """
#     Perform Synchrosqueezed Wavelet Transform on a 3D array of thermal data to extract high-resolution phase maps.
    
#     Parameters
#     ----------
#     T : numpy.ndarray
#         3D array of thermal data with dimensions (height, width, frames).
    
#     Returns
#     -------
#     sswt_amplitude : numpy.ndarray
#         Amplitude map obtained from the Synchrosqueezed Wavelet Transform.
#     sswt_phase : numpy.ndarray
#         Phase map obtained from the Synchrosqueezed Wavelet Transform.
    
#     Example
#     -------
#     >>> sswt_amplitude, sswt_phase = synchrosqueezed_wavelet_transform(T)
#     """
#     try:
#         import numpy as np
#         import ssqueezepy as ssq

#         # Compute the mean image over time
#         T_mean = np.mean(T, axis=2)
#         T_mean = T_mean.astype(np.float32)

#         # Flatten the image to apply 1D transform
#         signal = T_mean.flatten()

#         # Perform Synchrosqueezed Wavelet Transform
#         ssq_cwt, scales, _, _ = ssq.ssq_cwt(signal, wavelet='morlet')

#         # Compute amplitude and phase
#         amplitude = np.abs(ssq_cwt)
#         phase = np.angle(ssq_cwt)

#         # Reshape back to image dimensions
#         num_scales = amplitude.shape[0]
#         sswt_amplitude = amplitude.reshape(T_mean.shape + (num_scales,))
#         sswt_phase = phase.reshape(T_mean.shape + (num_scales,))

#         # For visualization, you might select a particular frequency slice
#         freq_index = amplitude.shape[1] // 2  # Select the central frequency component
#         amplitude_map = sswt_amplitude[:, :, freq_index]
#         phase_map = sswt_phase[:, :, freq_index]

#         return amplitude_map, phase_map

#     except Exception as e:
#         print(f"Error in synchrosqueezed_wavelet_transform: {e}")
#         return None, None


def dtw_clustering_defect_detection(T, n_clusters=4):
    """
    Use DTW-based clustering to detect defects.

    Parameters:
    T : numpy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    n_clusters : int
        Number of clusters to form.

    Returns:
    defect_map : numpy.ndarray
        2D map indicating cluster assignments.
    """
    try:
        from tslearn.clustering import TimeSeriesKMeans
        from tslearn.metrics import cdist_dtw
        height, width, frames = T.shape
        T_reshaped = T.reshape(-1, frames)

        # Perform clustering
        km_dtw = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", max_iter=10,n_jobs=-1)
        cluster_labels = km_dtw.fit_predict(T_reshaped)

        # Reshape cluster labels into image
        defect_map = cluster_labels.reshape(height, width)
        return defect_map
    except Exception as e:
        print(f"Error in dtw_clustering_defect_detection: {e}")
        return None

def frequency_ratio_imaging(T, fs, f_stim):
    """
    Compute frequency ratio imaging for defect detection.

    Parameters:
    T : numpy.ndarray
        3D array of thermal data with dimensions (height, width, frames).
    fs : float
        Sampling frequency.
    f_stim : float
        Stimulation frequency.

    Returns:
    defect_map : numpy.ndarray
        2D map highlighting potential defects.
    """
    try:
        height, width, frames = T.shape
        defect_map = np.zeros((height, width))

        freqs = np.fft.fftfreq(frames, d=1/fs)
        idx_fundamental = np.argmin(np.abs(freqs - f_stim))
        idx_harmonic = np.argmin(np.abs(freqs - 2*f_stim))

        for i in range(height):
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
        return defect_map
    except Exception as e:
        print(f"Error in frequency_ratio_imaging: {e}")
        return None
