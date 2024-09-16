import numpy as np
from scipy.ndimage import label





def lock_in_amplifier(temperature,time, frequency):
    def lock_in_amplifier(temperature, time, frequency):
        """
        Perform lock-in amplification on a 3D temperature dataset.
        Parameters:
        temperature (numpy.ndarray): A 3D array of temperature values with shape (rows, columns, samples).
        time (numpy.ndarray): A 1D array of time values corresponding to the temperature samples.
        frequency (float): The reference frequency for the lock-in amplifier.
        Returns:
        tuple: A tuple containing:
            - amplitude (numpy.ndarray): A 2D array of amplitude values with shape (rows, columns).
            - phase (numpy.ndarray): A 2D array of phase values with shape (rows, columns).
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
    The function calculates the centroid coordinates of the largest connected component in an image
    based on given amplitude and threshold values.
    
    :param amplitude: Amplitude is a measure of the magnitude of a signal or wave. In this context, it
    seems to be a 2D array representing the amplitude values of a signal
    :param threshold: The `threshold` parameter in the `calculate_centroid` function is used to
    determine the minimum value of the `amplitude` for which a pixel is considered part of the mask.
    Pixels with an `amplitude` greater than the `threshold` value will be included in the mask, while
    those
    :return: The function `calculate_centroid` returns the x and y coordinates of the centroid of the
    largest connected component in the input amplitude array based on the provided threshold.
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
    Generate a mask based on the amplitude and a given threshold.
    Parameters:
    amplitude (numpy.ndarray): The array of amplitude values.
    threshold (float): The threshold value to create the mask.
    Returns:
    numpy.ndarray: A boolean array where True indicates the amplitude is greater than the threshold.
    """
    
    mask = amplitude > threshold
    
    # Label connected components
    labeled_mask, num_features = label(mask)

    # Find the largest connected component
    largest_component = np.argmax(np.bincount(labeled_mask.flat)[1:]) + 1

    # Create a mask for the largest component
    mask = labeled_mask == largest_component
    return mask

import numpy as np
from scipy.signal import find_peaks
from scipy.signal import butter, filtfilt

def high_pass_filter(data, cutoff, fs, order=5):
    """
    Apply a high-pass filter to the data.

    Parameters:
    - data: The input data to be filtered.
    - cutoff: The cutoff frequency of the filter.
    - fs: The sampling frequency of the data.
    - order: The order of the filter.

    Returns:
    - The filtered data.
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    filtered_data = filtfilt(b, a, data)
    return filtered_data

# atpp/lock_in_imaging.py

import numpy as np

def find_se_frames(T, threshold, cutoff, fs, order):
    """
    Find the start and end frames of a signal based on a threshold and cutoff frequency.

    Parameters:
    - T: Temperature data (3D array: height x width x time)
    - threshold: Threshold value to determine significant frames
    - cutoff: Cutoff frequency for filtering
    - fs: Sampling frequency
    - order: Order of the filter

    Returns:
    - start_frame: The starting frame index
    - end_frame: The ending frame index
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