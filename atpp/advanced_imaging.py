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
                        phase[i, j + 1]
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



