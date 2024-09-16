import numpy as np
from scipy.ndimage import label





def lock_in_amplifier(flir_video, frequency):
    """
    The `lock_in_amplifier` function calculates the amplitude and phase of a signal in a FLIR video
    using lock-in amplifier technique.
    
    :param flir_video: The `lock_in_amplifier` function takes in a FLIR video object, a frequency value,
    and start and end frame indices as input parameters. It calculates the amplitude and phase values
    using lock-in amplifier technique for each pixel in the video frames within the specified frame
    range
    :param frequency: Frequency is the frequency of the signal in hertz that you want to lock in to
    extract the amplitude and phase information from the FLIR video data
    :return: The function `lock_in_amplifier` is returning two NumPy arrays - `amplitude` and `phase`.
    """
    temperature = flir_video.temperature
    time = flir_video.time

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

def mask_data(amplitude, phase, threshold):
    """
    The `mask_data` function applies a threshold to the amplitude data and returns a masked version
    of the amplitude and phase data.
    
    :param amplitude: The `amplitude` parameter is a 2D NumPy array representing the amplitude values
    of a signal.
    :param phase: The `phase` parameter is a 2D NumPy array representing the phase values of a signal.
    :param threshold: The `threshold` parameter is a float value that represents the threshold for
    masking the amplitude data.
    :return: The `mask_data` function returns two 2D NumPy arrays - `masked_amplitude` and `masked_phase`
    that contain the masked amplitude and phase data based on the provided threshold.
    """
    mask = amplitude > threshold
    masked_amplitude = np.where(mask, amplitude, np.nan)
    masked_phase = np.where(mask, phase, np.nan)
    
    return masked_amplitude, masked_phase