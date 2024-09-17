"""
This module provides functions for analyzing pulsed diffusivity data, including Gaussian fitting and centroid calculation.

Functions:
    - gaussian: Define a Gaussian function for curve fitting.
    - calculate_centroid: Calculate the centroid of the peak frame and create a binary mask.

Example usage:
    >>> from pulsed_diffusivity import gaussian, calculate_centroid
    >>> x = np.linspace(-10, 10, 100)
    >>> A, x0, sigma = 1, 0, 1
    >>> y = gaussian(x, A, x0, sigma)
    >>> peak_frame = np.random.rand(100, 100)
    >>> ambient_temp = np.random.rand(100, 100)
    >>> radius = 5
    >>> resolution = 0.1
    >>> centroid, mask_bin_expanded = calculate_centroid(peak_frame, ambient_temp, radius, resolution)
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression

n_frame_tamb = 100  # Constant for the number of ambient frames

def gaussian(x, A, x0, sigma):
    """
    Define a Gaussian function for curve fitting.

    :param x: The input data points.
    :type x: numpy.ndarray
    :param A: The amplitude of the Gaussian.
    :type A: float
    :param x0: The mean (center) of the Gaussian.
    :type x0: float
    :param sigma: The standard deviation (width) of the Gaussian.
    :type sigma: float
    :return: The Gaussian function evaluated at x.
    :rtype: numpy.ndarray

    Example:
        >>> x = np.linspace(-10, 10, 100)
        >>> A, x0, sigma = 1, 0, 1
        >>> y = gaussian(x, A, x0, sigma)
    """
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def calculate_centroid(peak_frame, ambient_temp, radius, resolution):
    """
    Calculate the centroid of the peak frame and create a binary mask.

    :param peak_frame: The frame with the peak temperature values.
    :type peak_frame: numpy.ndarray
    :param ambient_temp: The ambient temperature frame.
    :type ambient_temp: numpy.ndarray
    :param radius: The radius around the centroid to create the binary mask.
    :type radius: float
    :param resolution: The spatial resolution of the data.
    :type resolution: float
    :return: The coordinates of the centroid and the expanded binary mask.
    :rtype: tuple(tuple(float, float), numpy.ndarray)

    Example:
        >>> peak_frame = np.random.rand(100, 100)
        >>> ambient_temp = np.random.rand(100, 100)
        >>> radius = 5
        >>> resolution = 0.1
        >>> centroid, mask_bin_expanded = calculate_centroid(peak_frame, ambient_temp, radius, resolution)
    """
    mask_centroid = (peak_frame - ambient_temp) > 0.3 * (np.max(peak_frame) - ambient_temp)
    centroid = ndimage.center_of_mass(mask_centroid)

    r = int(radius / resolution)
    x, y = np.meshgrid(np.arange(peak_frame.shape[1]), np.arange(peak_frame.shape[0]))
    dist = np.sqrt((x - centroid[1]) ** 2 + (y - centroid[0]) ** 2)
    mask_bin = dist <= r
    mask_bin_expanded = np.expand_dims(mask_bin, axis=-1)

    return centroid, mask_bin_expanded

def apply_mask_and_get_ambient_temp(T, mask_bin_expanded):
    """
    Apply a binary mask to the temperature data and calculate the ambient temperature.

    :param T: 3D array of temperature data with dimensions (height, width, frames).
    :type T: numpy.ndarray
    :param mask_bin_expanded: Binary mask to apply to the temperature data.
    :type mask_bin_expanded: numpy.ndarray
    :return: Masked temperature data and the calculated ambient temperature.
    :rtype: tuple(numpy.ndarray, float)

    Example:
        >>> T = np.random.rand(100, 100, 1000)  # Example 3D array
        >>> mask_bin_expanded = np.random.randint(0, 2, (100, 100, 1))  # Example binary mask
        >>> masked_frame, ambient_temp = apply_mask_and_get_ambient_temp(T, mask_bin_expanded)
    """
    masked_frame = np.where(mask_bin_expanded, T, np.nan)
    ambient_temp = np.nanmean(masked_frame[:, :, :n_frame_tamb], axis=(0, 1, 2))
    
    return masked_frame, ambient_temp


def fit_gaussian_to_frames(masked_frame, centroid, start_frame, end_frame, resolution, ambient_temp, plot=False):
    """
    Fit a Gaussian function to the temperature data along a line through the centroid for each frame.

    :param masked_frame: 3D array of masked temperature data with dimensions (height, width, frames).
    :type masked_frame: numpy.ndarray
    :param centroid: Coordinates of the centroid (y, x).
    :type centroid: tuple(float, float)
    :param start_frame: Index of the start frame for fitting.
    :type start_frame: int
    :param end_frame: Index of the end frame for fitting.
    :type end_frame: int
    :param resolution: Spatial resolution of the data.
    :type resolution: float
    :param ambient_temp: Ambient temperature value to subtract from the data.
    :type ambient_temp: float
    :param plot: If True, plots the Gaussian fit for the first few frames, defaults to False.
    :type plot: bool, optional
    :return: Array of fitted Gaussian sigma values for each frame.
    :rtype: numpy.ndarray

    Example:
        >>> masked_frame = np.random.rand(100, 100, 1000)  # Example 3D array
        >>> centroid = (50, 50)  # Example centroid
        >>> start_frame = 10  # Example start frame
        >>> end_frame = 20  # Example end frame
        >>> resolution = 0.1  # Example resolution
        >>> ambient_temp = 300  # Example ambient temperature
        >>> sigma_values = fit_gaussian_to_frames(masked_frame, centroid, start_frame, end_frame, resolution, ambient_temp, plot=True)
    """
    sigma_values = []

    for i in range(start_frame, end_frame):
        line = masked_frame[int(centroid[0]), :, i] - ambient_temp
        x = np.arange(len(line))

        valid_indices = ~np.isnan(line)
        line = line[valid_indices]
        x = x[valid_indices]
        x = x * resolution

        if len(line) == 0:
            sigma_values.append(np.nan)
            continue

        x0 = centroid[1] * resolution
        A = np.max(line)
        p0 = [A, x0, np.std(x)]

        try:
            popt, _ = curve_fit(gaussian, x, line, p0=p0)

            # If plot=True, plot the fit for the first few frames where conditions are met
            if plot and i < start_frame + 15 and i > start_frame:
                if len(sigma_values) > 0 and popt[2] - sigma_values[-1] > 0:
                    xplot = np.linspace(x[0], x[-1], 1000)
                    plt.plot(xplot, gaussian(xplot, *popt), '-', label=f'Fit: Frame {i}', color='black', linewidth=1.3)
                    plt.plot(x, line, 'o', markersize=2, label=f'Frame {i}', color='grey', alpha=0.7)

            sigma_values.append(popt[2])
        except RuntimeError as e:
            print(f"Could not fit a Gaussian to frame {i}: {e}")
            sigma_values.append(np.nan)

    if plot:
        plt.xlabel('x (mm)')
        plt.ylabel(r'$T_{inc}$ (K)')
        # plt.legend()
        plt.show()

    return np.array(sigma_values)


def plot_line_through_centroid(fvd, num_frames, resolution, radius, plot=False):
    """
    Plot a line through the centroid of the peak frame and fit a Gaussian to the temperature data.

    :param fvd: FLIR video data object containing temperature data and time information.
    :type fvd: object
    :param num_frames: Number of frames to analyze after the peak frame.
    :type num_frames: int
    :param resolution: Spatial resolution of the data.
    :type resolution: float
    :param radius: Radius around the centroid to create the binary mask.
    :type radius: float
    :param plot: If True, plots the Gaussian fit for the first few frames, defaults to False.
    :type plot: bool, optional
    :return: Array of fitted Gaussian sigma values and corresponding times.
    :rtype: tuple(numpy.ndarray, numpy.ndarray)

    Example:
        >>> fvd = ...  # FLIR video data object
        >>> num_frames = 50  # Example number of frames
        >>> resolution = 0.1  # Example resolution
        >>> radius = 5  # Example radius
        >>> sigma_values, sigma_times = plot_line_through_centroid(fvd, num_frames, resolution, radius, plot=True)
    """
    peak_frame_idx = np.argmax(np.mean(fvd.Temp, axis=(0, 1)))
    start_frame = peak_frame_idx + 1
    end_frame = start_frame + num_frames

    T = fvd.Temp
    peak_frame = T[..., peak_frame_idx]

    start_frame = peak_frame_idx - int(0.1 * fvd.framerate)

    centroid, mask_bin_expanded = calculate_centroid(peak_frame, ambient_temp=np.nan, radius=radius, resolution=resolution)
    masked_frame, ambient_temp = apply_mask_and_get_ambient_temp(T, mask_bin_expanded)

    # Recalculate the centroid and mask after adjusting for ambient temperature
    centroid, mask_bin_expanded = calculate_centroid(peak_frame, ambient_temp, radius, resolution)
    masked_frame, _ = apply_mask_and_get_ambient_temp(T, mask_bin_expanded)

    # Fit the Gaussian and optionally plot
    sigma_values = fit_gaussian_to_frames(masked_frame, centroid, start_frame, end_frame, resolution, ambient_temp, plot=plot)
    sigma_times = fvd.time[start_frame:end_frame]

    sigma_times = np.array(sigma_times) - sigma_times[0]  # Normalize time to start from zero

    return sigma_values, sigma_times


def calculate_diffusivity(sigma_values, sigma_times, diff_tresh=0.2, plot=False):
    """
    Calculate the thermal diffusivity from the fitted Gaussian sigma values over time.

    :param sigma_values: Array of fitted Gaussian sigma values.
    :type sigma_values: numpy.ndarray
    :param sigma_times: Array of corresponding times for the sigma values.
    :type sigma_times: numpy.ndarray
    :param diff_tresh: Threshold for the second derivative to adjust the number of frames dynamically, defaults to 0.2.
    :type diff_tresh: float, optional
    :param plot: If True, plots the linear fit of sigma squared over time, defaults to False.
    :type plot: bool, optional
    :raises ValueError: If there are not enough data points to calculate diffusivity.
    :raises ValueError: If there are not enough valid points for linear regression.
    :raises ValueError: If there is an error in linear regression.
    :return: Calculated diffusivity, slope, intercept, sigma values, sigma times, and sigma squared values.
    :rtype: tuple(float, float, float, numpy.ndarray, numpy.ndarray, numpy.ndarray)

    Example:
        >>> sigma_values = np.random.rand(50)  # Example sigma values
        >>> sigma_times = np.linspace(0, 10, 50)  # Example sigma times
        >>> diffusivity, slope, intercept, sigma_values, sigma_times, sigma_square = calculate_diffusivity(sigma_values, sigma_times, plot=True)
    """
    if len(sigma_values) < 2 or len(sigma_times) < 2:
        raise ValueError("Not enough data points to calculate diffusivity.")
    
    # Compute the squared sigma values for diffusivity calculation
    sigma_square = sigma_values ** 2

    # First, consider the second derivative to adjust num_frames dynamically
    second_diff = np.diff(np.diff(sigma_square))
    try:
        # Find the first index where the second derivative exceeds the threshold
        idx = np.where(second_diff[3:] > diff_tresh)[0][0] + 3
        sigma_values = sigma_values[:idx]
        sigma_times = sigma_times[:idx]
        sigma_square = sigma_square[:idx]
    except IndexError:
        # If no such index is found, use the original arrays
        pass
    
    # Adjust the start frame based on the first derivative (negative slope condition)
    first_diff = np.diff(sigma_square)
    start_idx = 0
    for i in range(10):
        if first_diff[i] < 0:
            start_idx = i + 1
            break
    
    sigma_values = sigma_values[start_idx:]
    sigma_times = sigma_times[start_idx:]
    sigma_square = sigma_square[start_idx:]

    # Remove NaN values
    valid_indices = ~np.isnan(sigma_values)
    sigma_values = sigma_values[valid_indices]
    sigma_times = sigma_times[valid_indices]
    sigma_square = sigma_square[valid_indices]

    if len(sigma_values) < 2 or len(sigma_times) < 2:
        raise ValueError("Not enough valid points for linear regression.")
    
    # Perform linear regression using RANSAC for robustness
    try:
        ransac = RANSACRegressor(estimator=LinearRegression())
        ransac.fit(sigma_times.reshape(-1, 1), sigma_square)
        slope = ransac.estimator_.coef_[0]
        intercept = ransac.estimator_.intercept_
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Error in linear regression: {e}")
    
    # Calculate diffusivity
    diffusivity = slope / 8

    # Optionally, plot the linear fit
    if plot:
        plt.plot(sigma_times, slope * sigma_times + intercept, '-', label='Fitted line', color='black')
        plt.plot(sigma_times, sigma_square, 'o', markersize=3, label='Original data', color='grey', fillstyle='none')
        plt.xlabel('Time (s)')
        plt.ylabel(r'$\sigma^2$ (mm$^2$)')
        plt.xlim(left=0)
        plt.ylim(bottom=0)
        plt.legend()
        plt.show()

    return diffusivity, slope, intercept, sigma_values, sigma_times, sigma_square
