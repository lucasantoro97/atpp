'''Under construction'''

def cut_data_x(amplitude, phase, centroid_y):
    """
    Get amplitude and phase maps and cut them along the x-axis at a fixed y-coordinate.
    amplitude and phase are 2D arrays.
    centroid_x and centroid_y are the coordinates of the centroid.
    Returns the cut amplitude and phase lines.
    """
    amplitude_cut = amplitude[centroid_y,:]
    phase_cut = phase[centroid_y,:]
    return amplitude_cut, phase_cut