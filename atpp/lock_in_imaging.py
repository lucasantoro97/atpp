"""
This module provides basic functions for lock-in imaging and related processing techniques.

Functions:
    - lock_in_amplifier: Perform lock-in amplifier processing on temperature data.
    - calculate_centroid: Calculate the centroid of the largest connected component in the amplitude data.
    - mask_data: Create a mask for the largest connected component in the amplitude data.
    - find_se_frames: Find the start and end frames based on a threshold after high-pass filtering.
    - desample: Resample the input data to a new sampling rate.
    - detrend: Detrend the temperature data using polynomial fitting.

Example usage:
    >>> from lock_in_imaging import lock_in_amplifier, calculate_centroid, mask_data, find_se_frames, desample, detrend
    >>> temperature = np.random.rand(100, 100, 1000)  # Example 3D array
    >>> time = np.linspace(0, 1, 1000)  # Example time array
    >>> frequency = 10.0  # Example frequency
    >>> amplitude, phase = lock_in_amplifier(temperature, time, frequency)
"""

import numpy as np
from scipy.ndimage import label
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks, savgol_filter, resample
from tqdm import tqdm
import logging

# Configure logging
logger = logging.getLogger(__name__)

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
    logger.info("Starting lock-in amplifier processing")
    rw, c, s = temperature.shape
    t = time

    cos_wave = 2 * np.cos(frequency * 2 * np.pi * t)
    sin_wave = 2 * np.sin(frequency * 2 * np.pi * t)

    amplitude = np.zeros((rw, c))
    phase = np.zeros((rw, c))

    for i in tqdm(range(rw), desc="Processing rows"):
        for j in range(c):
            temp = temperature[i, j, :]
            temp_mean = np.mean(temp)
            temp_demeaned = temp - temp_mean
            F = temp_demeaned * cos_wave
            G = temp_demeaned * sin_wave
            X = np.mean(F)
            Y = np.mean(G)
            amplitude[i, j] = np.sqrt(X ** 2 + Y ** 2)
            phase[i, j] = np.arctan2(Y, X)
    
    logger.info("Lock-in amplifier processing completed")
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
    logger.info("Calculating centroid of the largest connected component")
    mask = amplitude > threshold
    # Label connected components
    labeled_mask, num_features = label(mask)

    # Find the largest connected component
    largest_component = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1

    # Create a mask for the largest component
    largest_mask = labeled_mask == largest_component

    # Calculate centroid of the largest component
    centroid_x = np.sum(np.arange(amplitude.shape[1]) * np.sum(largest_mask, axis=0)) / np.sum(largest_mask)
    centroid_y = np.sum(np.arange(amplitude.shape[0]) * np.sum(largest_mask, axis=1)) / np.sum(largest_mask)
    
    logger.info(f"Centroid calculated at ({centroid_x}, {centroid_y})")
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
    logger.info("Creating mask for the largest connected component")
    mask = amplitude > threshold
    
    # Label connected components
    labeled_mask, num_features = label(mask)

    # Find the largest connected component
    largest_component = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1

    # Create a mask for the largest component
    mask = labeled_mask == largest_component
    logger.info("Mask created")
    return mask

def find_se_frames(T, fs):
    """
    Find the start and end frames automatically after detrending by detecting
    when the temperature starts rising based on the derivative of the mean temperature signal.

    :param T: 3D array of temperature data with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param fs: Sampling frequency of the signal (frames per second).
    :type fs: float
    :return: Start and end frames as integers.
    :rtype: tuple(int, int)

    Example:
        >>> T = np.random.rand(100, 100, 1000)  # Example 3D array
        >>> fs = 100.0  # Example sampling frequency
        >>> start_frame, end_frame = find_se_frames(T, fs)
    """
    logger.info("Starting to find start and end frames")
    T_mean = np.mean(T, axis=(0, 1))
    
    win_len = 5  # Window length for Savitzky-Golay filter

    # Smoothing the temperature data
    T_mean_smooth = savgol_filter(T_mean, window_length=win_len, polyorder=1)

    # Calculate the first derivative of the smoothed temperature signal
    dT = np.diff(T_mean_smooth) * fs

    # Dynamic threshold based on mean derivative
    threshold = np.min(T_mean_smooth) + 2 * np.mean(dT[:win_len]) 

    # Detecting the rising edge based on threshold
    rising_indices = np.where(dT > threshold)[0]

    # Check if any rising edges are detected
    if rising_indices.size == 0:
        logger.warning("Could not detect the start frame based on the derivative threshold.")
        start_frame = None
    else:
        start_frame = rising_indices[0] + 1  # Correct for diff offset
        # Then find the first local maximum after the start frame
        peaks, _ = find_peaks(T_mean_smooth[start_frame:], height=0)
        if peaks.size == 0:
            logger.warning("Could not detect the start frame based on peaks after threshold.")
            start_frame = None
        else:
            start_frame = start_frame + peaks[0]
            logger.info(f"Start frame detected at index {start_frame}")
        

    # End frame detection based on signal amplitude threshold
    T_min = np.min(T_mean)
    T_max = np.max(T_mean)
    T_range = T_max - T_min
    end_threshold_value = T_min + 0.9 * T_range
    above_end_threshold = T_mean >= end_threshold_value

    if not np.any(above_end_threshold):
        end_frame = None
        logger.warning("Could not detect the end frame based on the end threshold.")
    else:
        end_indices = np.where(above_end_threshold)[0]
        end_frame = end_indices[-1]
        logger.info(f"End frame detected at index {end_frame}")

    logger.info("Start and end frame detection completed")
    return start_frame, end_frame

def desample(T, time, fs, f_stim, ratio=10):
    """
    Resample the input 3D array `T` to a new sampling rate based on the stimulation frequency `f_stim`.

    :param T: 3D array representing the input data with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param time: 1D array representing the time vector corresponding to the frames in `T`.
    :type time: numpy.ndarray
    :param fs: Original sampling frequency of the input data.
    :type fs: float
    :param f_stim: Stimulation frequency used to determine the new sampling rate.
    :type f_stim: float
    :param ratio: Ratio of the new sampling rate to the stimulation frequency, default is 10.
    :type ratio: int, optional
    :return: Tuple containing the resampled 3D array, the new time vector, and the new sampling frequency.
    :rtype: tuple(numpy.ndarray, numpy.ndarray, float)

    Example:
        >>> T = np.random.rand(100, 100, 1000)
        >>> time = np.linspace(0, 10, 1000)
        >>> new_T, new_time, new_fs = desample(T, time, fs=100, f_stim=10)
    """
    logger.info("Starting resampling of data")
    # Calculate the new sampling frequency
    new_fs = ratio * f_stim
    logger.info(f"New sampling frequency: {new_fs} Hz")
    
    # Calculate the number of new frames
    num_frames_new = int(len(time) * new_fs / fs)
    logger.info(f"Number of new frames: {num_frames_new}")
    
    # Resample along the time axis using scipy's resample function
    new_T = resample(T, num_frames_new, axis=2)
    
    # Create a new time vector based on the new sampling frequency
    new_time = np.linspace(time[0], time[-1], num_frames_new)
    
    logger.info("Resampling completed")
    return new_T, new_time, new_fs

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
    logger.info("Starting detrending of temperature data")
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
    logger.info("Computing polynomial coefficients")
    p = V_pinv @ data.T  # Shape: (polynomial + 1, num_pixels)

    # Evaluate the fitted polynomial trend
    fitted_trend = V @ p  # Shape: (frames, num_pixels)

    # Detrend the data by subtracting the fitted trend
    detrended_data = data.T - fitted_trend  # Shape: (frames, num_pixels)

    # Reshape detrended data back to original dimensions
    detrended_T = detrended_data.T.reshape(height, width, frames)

    logger.info("Detrending completed")
    return detrended_T
