"""
This module provides basic functions for lock-in imaging and related processing techniques.

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
from scipy.signal import find_peaks

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



def find_se_frames(T):
    """
    Find the start and end frames automatically after detrending.

    :param T: 3D array of temperature data with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param polynomial: Degree of the polynomial for detrending, defaults to 2.
    :type polynomial: int, optional
    :return: Start and end frames.
    :rtype: tuple(int or None, int or None)

    Example:
        >>> T = np.random.rand(100, 100, 1000)  # Example 3D array
        >>> start_frame, end_frame = find_se_frames(T)
    """
    # Get dimensions
    height, width, frames = T.shape

    T_mean= np.mean(T, axis=(0, 1))

    # Calculate the range of the signal
    T_min = np.min(T_mean)
    T_max = np.max(T_mean)
    T_range = T_max - T_min

    # Automatically determine the thresholds
    # Start threshold at 10% above minimum
    start_threshold_value = T_min + 0.1 * T_range
    # End threshold at 90% of the maximum value
    end_threshold_value = T_min + 0.9 * T_range

    # Find where the signal exceeds the start threshold
    above_start_threshold = T_mean >= start_threshold_value
    # Find where the signal exceeds the end threshold
    above_end_threshold = T_mean >= end_threshold_value

    if not np.any(above_start_threshold):
        # Start threshold not crossed
        start_frame = None
    else:
        # Find the first index where the signal crosses the start threshold
        start_indices = np.where(above_start_threshold)[0]
        start_frame = start_indices[0]

        # Find the first local peak after the start frame
        peaks, _ = find_peaks(T_mean[start_frame:])
        if peaks.size > 0:
            start_frame += peaks[0]

    if not np.any(above_end_threshold):
        # End threshold not crossed
        end_frame = None
    else:
        # Find the last index where the signal is above the end threshold
        end_indices = np.where(above_end_threshold)[0]
        end_frame = end_indices[-1]

    return start_frame, end_frame




def detrend(T, time=None, polynomial=2):
    """
    Detrend the temperature data using polynomial fitting.

    :param T: 3D array of temperature data with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param time: 1D array of time points corresponding to the frames. If None, uses frame indices.
    :type time: numpy.ndarray, optional
    :param polynomial: Degree of the polynomial for fitting, defaults to 2.
    :type polynomial: int, optional
    :return: Detrended temperature data.
    :rtype: numpy.ndarray

    Example:
        >>> T = np.random.rand(100, 100, 1000)  # Example 3D array
        >>> detrended_T = detrend(T, polynomial=2)
    """
    # Get dimensions
    height, width, frames = T.shape
    
    if time is None:
        time = np.arange(frames)
        
    # Reshape the data to (pixels, frames)
    num_pixels = height * width
    data = T.reshape(num_pixels, frames)  # Shape: (num_pixels, frames)

    # Normalize time vector to improve numerical stability
    t_mean = np.mean(time)
    t_std = np.std(time)
    t_normalized = (time - t_mean) / t_std

    # Construct Vandermonde matrix for polynomial fitting
    V = np.vander(t_normalized, N=polynomial + 1)  # Shape: (frames, polynomial + 1)

    # Compute pseudoinverse of V
    V_pinv = np.linalg.pinv(V)  # Shape: (polynomial + 1, frames)

    # Compute polynomial coefficients for each pixel in a vectorized manner
    # Data is transposed to align dimensions for matrix multiplication
    p = V_pinv @ data.T  # Shape: (polynomial + 1, num_pixels)

    # Evaluate the fitted polynomial trend
    fitted_trend = V @ p  # Shape: (frames, num_pixels)

    # Detrend the data by subtracting the fitted trend
    detrended_data = data.T - fitted_trend  # Shape: (frames, num_pixels)

    # Reshape detrended data back to original dimensions
    detrended_T = detrended_data.T.reshape(height, width, frames)

    return detrended_T
