"""
This module provides functions for processing thermography data and FLIR video files.

The main function in this module is `process`, which processes the provided thermography data
or FLIR video file. It can apply filters, visualize the output data, and perform lock-in amplifier
processing if a frequency is provided.

Functions:
    - process: Process the provided thermography data or FLIR video file.
    - apply_noise_reduction: Apply noise reduction to the data using a Gaussian filter.
    - visualize_data: Visualize the processed thermography data.

Example usage:
    >>> from atpp.process import process
    >>> data = 'path/to/flir_video.mp4'
    >>> result = process(data, filter='noise_reduction', visualize=True, frequency=10.0)
"""

from .lock_in_imaging import lock_in_amplifier
from .fnv_class import FlirVideo
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def process(input_data, filter=None, visualize=False, frequency=None):
    """
    Process the provided thermography data or FLIR video file.

    :param input_data: The input thermography data or FLIR video file to be processed.
    :type input_data: str or numpy.ndarray
    :param filter: The filter to apply, such as 'noise_reduction', defaults to None
    :type filter: str, optional
    :param visualize: If True, visualizes the output data, defaults to False
    :type visualize: bool, optional
    :param frequency: Frequency for lock-in amplifier processing, defaults to None
    :type frequency: float, optional
    :return: Processed data (amplitude, phase, or filtered temperature data)
    :rtype: dict or numpy.ndarray

    Example:
        >>> data = 'path/to/flir_video.mp4'
        >>> result = process(data, filter='noise_reduction', visualize=True, frequency=10.0)
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
    """
    Apply noise reduction to the data using a Gaussian filter.

    :param data: The input data to be processed.
    :type data: numpy.ndarray
    :return: The data after noise reduction.
    :rtype: numpy.ndarray

    Example:
        >>> data = np.random.rand(100, 100)
        >>> reduced_noise_data = apply_noise_reduction(data)
    """
    # Convert data to numpy array if it's not already
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # Apply Gaussian filter for noise reduction
    sigma = 1  # Standard deviation for Gaussian kernel
    reduced_noise_data = gaussian_filter(data, sigma=sigma)

    return reduced_noise_data

def visualize_data(data):
    """
    Visualize the processed thermography data.

    :param data: The data to be visualized.
    :type data: numpy.ndarray

    Example:
        >>> data = np.random.rand(100, 100)
        >>> visualize_data(data)
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