import numpy as np
from scipy.signal import hilbert

def phase_coherence_imaging(T, fs, f_stim):
    """
    The function `phase_coherence_imaging` calculates the amplitude, phase, and phase coherence of a
    given 3D signal array `T` with respect to a specified stimulus frequency `f_stim` and sampling
    frequency `fs`.
    
    :param T: 3D numpy array (height, width, frames) containing the image data over time.
    :param fs: Sampling frequency of the signal.
    :param f_stim: Stimulus frequency for reference sine and cosine generation.
    :return: amplitude, phase, and phase_coherence arrays.
    """
    # Get dimensions
    height, width, frames = T.shape

    # Time vector
    t = np.arange(frames) / fs

    # Reference signals
    ref_sin = np.sin(2 * np.pi * f_stim * t)
    ref_cos = np.cos(2 * np.pi * f_stim * t)

    # Initialize arrays
    I = np.zeros((height, width))
    Q = np.zeros((height, width))
    amplitude = np.zeros((height, width))
    phase = np.zeros((height, width))

    # Compute In-phase (I) and Quadrature (Q) components
    for i in range(height):
        for j in range(width):
            signal = T[i, j, :]
            I[i, j] = np.sum(signal * ref_cos) / frames  # Normalize by number of frames
            Q[i, j] = np.sum(signal * ref_sin) / frames  # Normalize by number of frames

    # Calculate amplitude and phase
    amplitude = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)

    # Phase coherence computation
    phase_diff = np.zeros((height, width))
    for i in range(1, height-1):
        for j in range(1, width-1):
            neighbors = [
                phase[i-1, j],
                phase[i+1, j],
                phase[i, j-1],
                phase[i, j+1]
            ]
            phase_diff[i, j] = np.std([phase[i, j] - neighbor for neighbor in neighbors])

    # Normalize phase coherence map
    if np.max(phase_diff) > 0:
        phase_coherence = 1 - (phase_diff / np.max(phase_diff))
    else:
        phase_coherence = np.ones((height, width))  # Handle division by zero

    return amplitude, phase, phase_coherence


def synchronous_demodulation(T, fs, f_stim):
    """
    The function `synchronous_demodulation` performs synchronous demodulation on a 3D array of
    time-domain signals to extract amplitude and phase information using reference sinusoidal signals.
    
    :param T: It seems like the description of the parameter `T` is missing. Could you please provide
    more information about what `T` represents in the context of the `synchronous_demodulation`
    function?
    :param fs: The parameter `fs` represents the sampling frequency of the signal. It is the number of
    samples obtained in one second
    :param f_stim: f_stim is the frequency of the stimulus signal used for synchronous demodulation
    :return: The function `synchronous_demodulation` returns two arrays: `amplitude` and `phase`. The
    `amplitude` array contains the calculated amplitude values for each pixel in the input data, while
    the `phase` array contains the calculated phase values for each pixel.
    """
    height, width, frames = T.shape
    t = np.arange(frames) / fs

    # Reference signals
    ref_sin = np.sin(2 * np.pi * f_stim * t)
    ref_cos = np.cos(2 * np.pi * f_stim * t)

    # Initialize arrays
    I = np.zeros((height, width))
    Q = np.zeros((height, width))

    # Demodulate
    for i in range(height):
        for j in range(width):
            signal = T[i, j, :]
            I[i, j] = (2 / frames) * np.sum(signal * ref_cos)
            Q[i, j] = (2 / frames) * np.sum(signal * ref_sin)

    # Calculate amplitude and phase
    amplitude = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)

    return amplitude, phase


def hilbert_transform_analysis(T):
    """
    Perform Hilbert Transform analysis on a 3D array.

    This function computes the amplitude and phase of the analytic signal
    obtained from the Hilbert Transform of each pixel's time series in the 
    input 3D array.

    Parameters:
    T (numpy.ndarray): A 3D numpy array with shape (height, width, frames) 
                       representing the input data.

    Returns:
    tuple: A tuple containing two 3D numpy arrays:
        - amplitude (numpy.ndarray): The amplitude of the analytic signal 
                                     with the same shape as T.
        - phase (numpy.ndarray): The phase of the analytic signal with the 
                                 same shape as T.
    """

    height, width, frames = T.shape

    # Initialize arrays
    amplitude = np.zeros((height, width, frames))
    phase = np.zeros((height, width, frames))

    # Apply Hilbert Transform
    for i in range(height):
        for j in range(width):
            signal = T[i, j, :]
            analytic_signal = hilbert(signal)
            amplitude[i, j, :] = np.abs(analytic_signal)
            phase[i, j, :] = np.unwrap(np.angle(analytic_signal))

    return amplitude, phase


def thermal_signal_reconstruction(T, order=5):
    """
    The function `thermal_signal_reconstruction` reconstructs thermal signals using polynomial fitting
    on the logarithm of the input signal.
    
    :param T: It seems like you were about to provide some information about the parameter `T` in the
    `thermal_signal_reconstruction` function. Could you please provide more details or specify the shape
    and type of the `T` parameter so that I can assist you further with the function?
    :param order: The `order` parameter in the `thermal_signal_reconstruction` function represents the
    degree of the polynomial used for fitting the signal data. A higher order polynomial can capture
    more complex patterns in the data but may also be more prone to overfitting. You can adjust the
    `order` parameter to control, defaults to 5 (optional)
    :return: The function `thermal_signal_reconstruction` returns the reconstructed thermal signal array
    `T_reconstructed`.
    """
    height, width, frames = T.shape
    log_time = np.log(np.arange(1, frames + 1))

    # Initialize reconstructed signal array
    T_reconstructed = np.zeros_like(T)

    for i in range(height):
        for j in range(width):
            signal = T[i, j, :]
            log_signal = np.log(signal + np.finfo(float).eps)  # Avoid log(0)
            # Polynomial fitting
            coeffs = np.polyfit(log_time, log_signal, order)
            # Reconstruct signal
            log_signal_fit = np.polyval(coeffs, log_time)
            T_reconstructed[i, j, :] = np.exp(log_signal_fit)

    return T_reconstructed


def modulated_thermography(T, fs, f_stim, harmonics=[2, 3]):
    """
    The function `modulated_thermography` performs demodulation on a thermal image sequence using
    modulated thermography technique to extract amplitude and phase information for specified harmonics.
    
    :param T: It seems like you were about to provide some information about the parameter T in the
    `modulated_thermography` function. Could you please specify what T represents in this context?
    :param fs: The parameter `fs` in the `modulated_thermography` function represents the sampling
    frequency of the thermal data `T`. It is used to calculate the time vector `t` based on the number
    of frames in the data. If you have any more questions or need further clarification, feel
    :param f_stim: The `f_stim` parameter in the `modulated_thermography` function represents the
    frequency of the stimulus signal used in modulated thermography. This frequency is used to generate
    reference signals for demodulation at different harmonics
    :param harmonics: The `harmonics` parameter in the `modulated_thermography` function is a list that
    specifies the harmonics for which you want to analyze the modulated thermography data. By default,
    the function calculates the amplitude and phase for the 2nd and 3rd harmonics
    :return: The function `modulated_thermography` returns two dictionaries: `amplitude` and `phase`.
    The `amplitude` dictionary contains the calculated amplitudes for each harmonic specified in the
    `harmonics` list, while the `phase` dictionary contains the calculated phases for each harmonic.
    """
    height, width, frames = T.shape
    t = np.arange(frames) / fs

    # Initialize dictionaries to hold amplitude and phase for each harmonic
    amplitude = {}
    phase = {}

    for h in harmonics:
        # Reference signals for each harmonic
        ref_sin = np.sin(2 * np.pi * f_stim * h * t)
        ref_cos = np.cos(2 * np.pi * f_stim * h * t)

        I = np.zeros((height, width))
        Q = np.zeros((height, width))

        # Demodulate
        for i in range(height):
            for j in range(width):
                signal = T[i, j, :]
                I[i, j] = (2 / frames) * np.sum(signal * ref_cos)
                Q[i, j] = (2 / frames) * np.sum(signal * ref_sin)

        # Store amplitude and phase
        amplitude[h] = np.sqrt(I**2 + Q**2)
        phase[h] = np.arctan2(Q, I)

    return amplitude, phase
