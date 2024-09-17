"""
This module provides functions for lock-in imaging and related processing techniques.

Functions:
    - lock_in_amplifier: Perform lock-in amplifier processing on temperature data.
    - calculate_centroid: Calculate the centroid of the largest connected component in the amplitude data.
    - mask_data: Create a mask for the largest connected component in the amplitude data.
    - high_pass_filter: Apply a high-pass filter to the data.
    - find_se_frames: Find the start and end frames based on a threshold after high-pass filtering.

Example usage:
    >>> from lock_in_imaging import lock_in_amplifier, calculate_centroid, mask_data, high_pass_filter, find_se_frames
    >>> temperature = np.random.rand(100, 100, 1000)  # Example 3D array
    >>> time = np.linspace(0, 1, 1000)  # Example time array
    >>> frequency = 10.0  # Example frequency
    >>> amplitude, phase = lock_in_amplifier(temperature, time, frequency)
"""

import numpy as np
from scipy.ndimage import label
from scipy.signal import butter, filtfilt

def lock_in_amplifier(temperature, time, frequency):
    """
    Perform lock-in amplifier processing on temperature data.

    :param temperature: 3D array of temperature data with dimensions (height, width, frames).
    :type temperature: numpy.ndarray
    :param time: 1D array of time points corresponding to the frames.
    :type time: numpy.ndarray
    :param frequency: Frequency of the reference signal for lock-in processing.
    :type frequency: float
    :return: Amplitude and phase images.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)

    Example:
        >>> temperature = np.random.rand(100, 100, 1000)  # Example 3D array
        >>> time = np.linspace(0, 1, 1000)  # Example time array
        >>> frequency = 10.0  # Example frequency
        >>> amplitude, phase = lock_in_amplifier(temperature, time, frequency)
    """
    rw, c, s = temperature[:, :, :].shape
    t = time[:]

    cos_wave = 2 * np.cos(frequency * 2 * np.pi * t)
    sin_wave = 2 * np.sin(frequency * 2 * np.pi * t)

    amplitude = np.zeros((rw, c))
    phase = np.zeros((rw, c))

    for i in range(rw):
        for j in range(c):
            temp = temperature[i, j, :]
            F = (temp - np.mean(temp)) * cos_wave
            G = (temp - np.mean(temp)) * sin_wave
            X = np.mean(F)
            Y = np.mean(G)
            amplitude[i, j] = np.sqrt(X ** 2 + Y ** 2)
            phase[i, j] = np.arctan2(Y, X)
            
    return amplitude, phase

def calculate_centroid(amplitude, threshold):
    """
    Calculate the centroid of the largest connected component in the amplitude data.

    :param amplitude: 2D array of amplitude data.
    :type amplitude: numpy.ndarray
    :param threshold: Threshold value to create a binary mask.
    :type threshold: float
    :return: Coordinates of the centroid (x, y).
    :rtype: tuple(float, float)

    Example:
        >>> amplitude = np.random.rand(100, 100)  # Example amplitude data
        >>> threshold = 0.5  # Example threshold
        >>> centroid_x, centroid_y = calculate_centroid(amplitude, threshold)
    """
    mask = amplitude > threshold
    # Label connected components
    labeled_mask, num_features = label(mask)

    # Find the largest connected component
    largest_component = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1

    # Create a mask for the largest component
    largest_mask = labeled_mask == largest_component

    # Calculate centroid of the largest component
    centroid_x = np.sum(np.arange(amplitude.shape[1]) * largest_mask) / np.sum(largest_mask)
    centroid_y = np.sum(np.arange(amplitude.shape[0]).reshape(-1, 1) * largest_mask) / np.sum(largest_mask)
    
    return centroid_x, centroid_y

def mask_data(amplitude, threshold):
    """
    Create a mask for the largest connected component in the amplitude data.

    :param amplitude: 2D array of amplitude data.
    :type amplitude: numpy.ndarray
    :param threshold: Threshold value to create a binary mask.
    :type threshold: float
    :return: Binary mask of the largest connected component.
    :rtype: numpy.ndarray

    Example:
        >>> amplitude = np.random.rand(100, 100)  # Example amplitude data
        >>> threshold = 0.5  # Example threshold
        >>> mask = mask_data(amplitude, threshold)
    """
    mask = amplitude > threshold
    
    # Label connected components
    labeled_mask, num_features = label(mask)

    # Find the largest connected component
    largest_component = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1

    # Create a mask for the largest component
    mask = labeled_mask == largest_component
    return mask

def high_pass_filter(data, cutoff, fs, order=5):
    """
    Apply a high-pass filter to the data.

    :param data: 1D array of data to be filtered.
    :type data: numpy.ndarray
    :param cutoff: Cutoff frequency for the high-pass filter.
    :type cutoff: float
    :param fs: Sampling frequency of the data.
    :type fs: float
    :param order: Order of the filter, defaults to 5.
    :type order: int, optional
    :return: Filtered data.
    :rtype: numpy.ndarray

    Example:
        >>> data = np.random.rand(1000)  # Example data
        >>> cutoff = 0.1  # Example cutoff frequency
        >>> fs = 1000.0  # Example sampling frequency
        >>> filtered_data = high_pass_filter(data, cutoff, fs, order=5)
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

def find_se_frames(T, threshold, cutoff, fs, order):
    """
    Find the start and end frames based on a threshold after high-pass filtering.

    :param T: 3D array of temperature data with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param threshold: Threshold value to determine start and end frames.
    :type threshold: float
    :param cutoff: Cutoff frequency for the high-pass filter.
    :type cutoff: float
    :param fs: Sampling frequency of the data.
    :type fs: float
    :param order: Order of the high-pass filter.
    :type order: int
    :return: Start and end frames.
    :rtype: tuple(int, int)

    Example:
        >>> T = np.random.rand(100, 100, 1000)  # Example 3D array
        >>> threshold = 0.5  # Example threshold
        >>> cutoff = 0.1  # Example cutoff frequency
        >>> fs = 1000.0  # Example sampling frequency
        >>> order = 5  # Example filter order
        >>> start_frame, end_frame = find_se_frames(T, threshold, cutoff, fs, order)
    """
    # Flatten the spatial dimensions and take the maximum temperature signal
    T_max = np.max(np.max(T, axis=0), axis=0)

    # Apply high-pass filter
    filtered_temps = high_pass_filter(T_max, cutoff, fs, order)

    # Find the start and end frames based on the threshold
    above_threshold = filtered_temps > threshold
    start_frame = np.argmax(above_threshold)
    end_frame = len(filtered_temps) - np.argmax(above_threshold[::-1])

    return start_frame, end_frame