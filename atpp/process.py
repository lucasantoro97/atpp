"""Dummy func"""
from .lock_in_imaging import lock_in_amplifier
from .fnv_class import FlirVideo
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def process(input_data, filter=None, visualize=False, frequency=None):
    """
    Just a basic function - Process the provided thermography data.

    Args:
        input_data: The input thermography data or FLIR video file to be processed.
        filter: Optional. Apply a filter such as 'noise_reduction'.
        visualize: Optional. If True, visualizes the output data.
        frequency: Optional. Frequency for lock-in amplifier processing.

    Returns:
        Processed data (amplitude, phase, or filtered temperature data).
    """
    # Initialize the FlirVideo object from the input data file
    flir_video = FlirVideo(input_data)

    # If a frequency is provided, apply the lock-in amplifier
    if frequency:
        amplitude, phase = lock_in_amplifier(flir_video, frequency)
        processed_data = {'amplitude': amplitude, 'phase': phase}
    else:
        # Default processing of temperature data
        processed_data = flir_video.temperature

    # Apply noise reduction if specified
    if filter == "noise_reduction":
        processed_data = apply_noise_reduction(processed_data)

    # Visualize the processed data if requested
    if visualize:
        visualize_data(processed_data)

    return processed_data

def apply_noise_reduction(data):
    """Applies noise reduction to the thermography data."""
    # Convert data to numpy array if it's not already
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # Apply Gaussian filter for noise reduction
    sigma = 1  # Standard deviation for Gaussian kernel
    reduced_noise_data = gaussian_filter(data, sigma=sigma)

    return reduced_noise_data

def visualize_data(data):
    """
    Visualize thermography data using a heatmap.
    Parameters:
    data (array-like): The input data to visualize. It can be a 2D or 3D array. 
                       If it's a 3D array, only the first frame will be used.
    Returns:
    None: This function displays a plot and does not return any value.
    Notes:
    - The data is converted to a numpy array if it is not already one.
    - The data is displayed using a 'hot' colormap.
    - A colorbar is added to the plot to indicate the scale.
    - The plot includes titles and labels for the x and y axes.
    """
    
    # Convert data to numpy array if it's not already
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # Check if it's a 2D array or 3D array
    if len(data.shape) == 3:
        data = data[:, :, 0]  # Use the first frame if it's a 3D array

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Display the data as an image
    cax = ax.imshow(data, cmap='hot', interpolation='nearest')

    # Add a colorbar to show the scale
    fig.colorbar(cax)

    # Set the title and labels
    ax.set_title('Thermography Data Visualization')
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')

    # Show the plot
    plt.show()
