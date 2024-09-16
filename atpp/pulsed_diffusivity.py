import numpy as np
from scipy.optimize import curve_fit
from scipy import ndimage
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression

'''STILL TO BE REVIEWED AND REFACTORED'''

n_frame_tamb = 100  # Constant for the number of ambient frames

def gaussian(x, A, x0, sigma):
    """
    Gaussian function used for curve fitting.

    Parameters:
    - x: array-like, input data
    - A: float, amplitude
    - x0: float, mean (center of the peak)
    - sigma: float, standard deviation (width of the peak)

    Returns:
    - Gaussian function evaluated at x.
    """
    return A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def calculate_centroid(peak_frame, ambient_temp, radius, resolution):
    """
    Calculates the centroid of the area where the temperature increase is greater than 30% 
    of the maximum temperature rise.

    Parameters:
    - peak_frame: 2D array of the peak temperature frame.
    - ambient_temp: float, ambient temperature.
    - radius: float, radius in mm.
    - resolution: float, spatial resolution in mm/px.

    Returns:
    - centroid: tuple, coordinates of the centroid.
    - mask_bin_expanded: 3D binary mask with the same shape as the frame stack.
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
    Applies a binary mask to a 3D temperature array and calculates the ambient temperature.

    Parameters:
    - T: 3D array of temperature values (height, width, time).
    - mask_bin_expanded: 3D binary mask.

    Returns:
    - masked_frame: masked 3D temperature array.
    - ambient_temp: float, mean ambient temperature.
    """
    masked_frame = np.where(mask_bin_expanded, T, np.nan)
    ambient_temp = np.nanmean(masked_frame[:, :, :n_frame_tamb], axis=(0, 1, 2))
    
    return masked_frame, ambient_temp


def fit_gaussian_to_frames(masked_frame, centroid, start_frame, end_frame, resolution, ambient_temp, plot=False):
    """
    Fits a Gaussian curve to temperature data along the x-axis of the masked frames and returns the fitted
    sigma values and corresponding times. Optionally plots the Gaussian fits if plot=True.
    
    Parameters:
    - masked_frame: 3D array of masked temperature data.
    - centroid: tuple, coordinates of the centroid.
    - start_frame: int, starting frame index.
    - end_frame: int, ending frame index.
    - resolution: float, spatial resolution in mm/px.
    - ambient_temp: float, ambient temperature.
    - plot: bool, whether to plot the Gaussian fits. Defaults to False.
    
    Returns:
    - sigma: 1D array of fitted sigma values.
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
    Main function to analyze the temperature evolution along the centroid of the 
    temperature distribution. It fits Gaussian curves and extracts sigma values.

    Parameters:
    - fvd: object containing temperature and time data.
    - num_frames: int, number of frames to analyze.
    - resolution: float, spatial resolution in mm/px.
    - radius: float, radius of the area of interest in mm.
    - plot: bool, whether to plot the Gaussian fits.

    Returns:
    - sigma_values: 1D array of fitted sigma values.
    - sigma_times: 1D array of times corresponding to the fitted sigma values.
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
    Calculate the diffusivity from the given sigma and sigma_times values, and return additional 
    information for plotting (slope, intercept, and processed sigma_values and sigma_times).

    Parameters:
    - sigma_values: 1D array, fitted sigma values (from Gaussian fits).
    - sigma_times: 1D array, times corresponding to the sigma values.
    - diff_tresh: float, threshold for recalculating the number of frames based on the second derivative.
    - plot: bool, whether to plot the Gaussian fits.

    Returns:
    - diffusivity: float, calculated diffusivity value.
    - slope: float, slope of the linear fit on sigma_squared values.
    - intercept: float, intercept of the linear fit.
    - sigma_values: 1D array, processed sigma values after adjustments.
    - sigma_times: 1D array, processed sigma times after adjustments.
    - sigma_square: 1D array, processed sigma squared values.
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
