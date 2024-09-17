"""
This module provides advanced imaging techniques for thermography data analysis.

Functions:
    - phase_coherence_imaging: Perform phase coherence imaging on a 3D array of time-domain signals.
    - synchronous_demodulation: Perform synchronous demodulation on a 3D array of time-domain signals.
    - hilbert_transform_analysis: Perform Hilbert transform analysis on a 3D array of time-domain signals.
    - thermal_signal_reconstruction: Reconstruct thermal signals using polynomial fitting.
    - modulated_thermography: Perform modulated thermography analysis on a 3D array of time-domain signals.

Example usage:
    >>> from advanced_imaging import phase_coherence_imaging
    >>> T = np.random.rand(100, 100, 1000)  # Example 3D array
    >>> fs = 1000.0  # Example sampling frequency
    >>> f_stim = 10.0  # Example stimulus frequency
    >>> result = phase_coherence_imaging(T, fs, f_stim)
"""

import numpy as np
from scipy.signal import hilbert

def phase_coherence_imaging(T, fs, f_stim):
    """
    Perform phase coherence imaging on a 3D array of time-domain signals to extract phase coherence information.

    :param T: A 3D numpy array of time-domain signals with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param fs: The sampling frequency of the signal. It is the number of samples obtained in one second.
    :type fs: float
    :param f_stim: The frequency of the stimulus signal used for phase coherence imaging.
    :type f_stim: float
    :return: A tuple containing amplitude, phase, and phase coherence images.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, numpy.ndarray)

    Example:
        >>> T = np.random.rand(100, 100, 1000)  # Example 3D array
        >>> fs = 1000.0  # Example sampling frequency
        >>> f_stim = 10.0  # Example stimulus frequency
        >>> amplitude, phase, phase_coherence = phase_coherence_imaging(T, fs, f_stim)
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
    Perform synchronous demodulation on a 3D array of time-domain signals to extract amplitude and phase information.

    :param T: A 3D numpy array of time-domain signals with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param fs: The sampling frequency of the signal. It is the number of samples obtained in one second.
    :type fs: float
    :param f_stim: The frequency of the stimulus signal used for synchronous demodulation.
    :type f_stim: float
    :return: A tuple containing amplitude and phase images.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)

    Example:
        >>> T = np.random.rand(100, 100, 1000)  # Example 3D array
        >>> fs = 1000.0  # Example sampling frequency
        >>> f_stim = 10.0  # Example stimulus frequency
        >>> amplitude, phase = synchronous_demodulation(T, fs, f_stim)
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
    Perform Hilbert transform analysis on a 3D array of time-domain signals to extract amplitude and phase information.

    :param T: A 3D numpy array of time-domain signals with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :return: A tuple containing amplitude and phase images.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)

    Example:
        >>> T = np.random.rand(100, 100, 1000)  # Example 3D array
        >>> amplitude, phase = hilbert_transform_analysis(T)
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
    Reconstruct thermal signals using polynomial fitting.

    :param T: A 3D numpy array of time-domain signals with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param order: The order of the polynomial for fitting, defaults to 5.
    :type order: int, optional
    :return: A 3D numpy array of reconstructed thermal signals.
    :rtype: numpy.ndarray

    Example:
        >>> T = np.random.rand(100, 100, 1000)  # Example 3D array
        >>> T_reconstructed = thermal_signal_reconstruction(T, order=5)
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
    Perform modulated thermography analysis on a 3D array of time-domain signals.

    :param T: A 3D numpy array of time-domain signals with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param fs: The sampling frequency of the signal. It is the number of samples obtained in one second.
    :type fs: float
    :param f_stim: The frequency of the stimulus signal used for modulated thermography.
    :type f_stim: float
    :param harmonics: A list of harmonics to analyze, defaults to [2, 3].
    :type harmonics: list, optional
    :return: A tuple containing dictionaries of amplitude and phase images for each harmonic.
    :rtype: tuple(dict, dict)

    Example:
        >>> T = np.random.rand(100, 100, 1000)  # Example 3D array
        >>> fs = 1000.0  # Example sampling frequency
        >>> f_stim = 10.0  # Example stimulus frequency
        >>> harmonics = [2, 3]  # Example harmonics
        >>> amplitude, phase = modulated_thermography(T, fs, f_stim, harmonics)
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