import numpy as np
from scipy.signal import hilbert

def phase_coherence_imaging(T, fs, f_stim):
    """
    Compute the phase coherence imaging of a given 3D time series data.

    Parameters:
    T (numpy.ndarray): 3D array of shape (height, width, frames) representing the time series data.
    fs (float): Sampling frequency of the time series data.
    f_stim (float): Stimulation frequency.

    Returns:
    tuple: A tuple containing:
        - amplitude (numpy.ndarray): 2D array of shape (height, width) representing the amplitude of the signal.
        - phase (numpy.ndarray): 2D array of shape (height, width) representing the phase of the signal.
        - phase_coherence (numpy.ndarray): 2D array of shape (height, width) representing the phase coherence map.
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
    Perform synchronous demodulation on a 3D array of time-series data.

    Parameters:
    T (numpy.ndarray): A 3D array of shape (height, width, frames) representing the time-series data.
    fs (float): The sampling frequency of the time-series data.
    f_stim (float): The frequency of the stimulus signal.

    Returns:
    tuple: A tuple containing two 2D arrays:
        - amplitude (numpy.ndarray): The amplitude of the demodulated signal, of shape (height, width).
        - phase (numpy.ndarray): The phase of the demodulated signal, of shape (height, width).
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
    T (numpy.ndarray): A 3D numpy array with shape (height, width, frames) representing the input data.

    Returns:
    tuple: A tuple containing two 3D numpy arrays:
        - amplitude (numpy.ndarray): The amplitude of the analytic signal with the same shape as T.
        - phase (numpy.ndarray): The phase of the analytic signal with the same shape as T.
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
    Reconstructs the thermal signal using polynomial fitting in the logarithmic domain.

    Parameters:
    T (numpy.ndarray): A 3D array of shape (height, width, frames) representing the thermal signal over time.
    order (int, optional): The order of the polynomial to fit. Default is 5.

    Returns:
    numpy.ndarray: A 3D array of the same shape as T containing the reconstructed thermal signal.
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
    Perform modulated thermography analysis on a 3D thermal data array.

    Parameters:
    T (numpy.ndarray): 3D array of thermal data with shape (height, width, frames).
    fs (float): Sampling frequency of the thermal data.
    f_stim (float): Stimulation frequency.
    harmonics (list of int, optional): List of harmonics to analyze. Default is [2, 3].

    Returns:
    tuple: Two dictionaries containing amplitude and phase for each harmonic.
        - amplitude (dict): Amplitude of the thermal response for each harmonic.
        - phase (dict): Phase of the thermal response for each harmonic.
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
