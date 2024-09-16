import numpy as np
from scipy.signal import hilbert

def phase_coherence_imaging(T, fs, f_stim):
    """
    The function `phase_coherence_imaging` calculates the amplitude, phase, and phase coherence of a
    given 3D signal array `T` with respect to a specified stimulus frequency `f_stim` and sampling
    frequency `fs`.
    
    :param T: The parameter `T` in the `phase_coherence_imaging` function represents a 3D numpy array
    that contains the image data over time. The dimensions of this array are `(height, width, frames)`,
    where `height` and `width` represent the spatial dimensions of the image,
    :param fs: The parameter `fs` in the `phase_coherence_imaging` function represents the sampling
    frequency of the signal. It is used to create the time vector `t` based on the number of frames in
    the input data `T`. This sampling frequency is essential for generating the reference sinusoidal
    signals at
    :param f_stim: The `f_stim` parameter in the `phase_coherence_imaging` function represents the
    frequency of the stimulus signal that is being used in the computation. This frequency is used to
    generate reference sinusoidal signals (sine and cosine) for comparison with the input signal in
    order to calculate the phase
    :return: The function `phase_coherence_imaging` returns three arrays: `amplitude`, `phase`, and
    `phase_coherence`.
    """
    # Get dimensions
    height, width, frames = T.shape

    # Time vector
    t = np.arange(frames) / fs

    # Reference signals
    ref_sin = np.sin(2 * np.pi * f_stim * t)
    ref_cos = np.cos(2 * np.pi * f_stim * t)

    # Initialize arrays
    I = np.zeros((height, width))
    Q = np.zeros((height, width))
    amplitude = np.zeros((height, width))
    phase = np.zeros((height, width))

    # Compute In-phase (I) and Quadrature (Q) components
    for i in range(height):
        for j in range(width):
            signal = T[i, j, :]
            I[i, j] = np.sum(signal * ref_cos)
            Q[i, j] = np.sum(signal * ref_sin)

    # Calculate amplitude and phase
    amplitude = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)

    # Phase coherence computation
    # Compute phase difference with neighboring pixels
    phase_diff = np.zeros((height, width))
    for i in range(1, height-1):
        for j in range(1, width-1):
            neighbors = [
                phase[i-1, j],
                phase[i+1, j],
                phase[i, j-1],
                phase[i, j+1]
            ]
            phase_diff[i, j] = np.std([phase[i, j] - neighbor for neighbor in neighbors])

    # Normalize phase coherence map
    phase_coherence = 1 - (phase_diff / np.max(phase_diff))

    return amplitude, phase, phase_coherence

def synchronous_demodulation(T, fs, f_stim):
    """
    The function `synchronous_demodulation` performs synchronous demodulation on a 3D array of
    time-domain signals to extract amplitude and phase information using reference sinusoidal signals.
    
    :param T: It seems like the description of the parameter `T` is missing. Could you please provide
    more information about what `T` represents in the context of the `synchronous_demodulation`
    function?
    :param fs: The parameter `fs` represents the sampling frequency of the signal. It is the number of
    samples obtained in one second
    :param f_stim: f_stim is the frequency of the stimulus signal used for synchronous demodulation
    :return: The function `synchronous_demodulation` returns two arrays: `amplitude` and `phase`. The
    `amplitude` array contains the calculated amplitude values for each pixel in the input data, while
    the `phase` array contains the calculated phase values for each pixel.
    """
    height, width, frames = T.shape
    t = np.arange(frames) / fs

    # Reference signals
    ref_sin = np.sin(2 * np.pi * f_stim * t)
    ref_cos = np.cos(2 * np.pi * f_stim * t)

    # Initialize arrays
    I = np.zeros((height, width))
    Q = np.zeros((height, width))

    # Demodulate
    for i in range(height):
        for j in range(width):
            signal = T[i, j, :]
            I[i, j] = (2 / frames) * np.sum(signal * ref_cos)
            Q[i, j] = (2 / frames) * np.sum(signal * ref_sin)

    # Calculate amplitude and phase
    amplitude = np.sqrt(I**2 + Q**2)
    phase = np.arctan2(Q, I)

    return amplitude, phase


def hilbert_transform_analysis(T):
    """
    The function `hilbert_transform_analysis` computes the mean amplitude and phase over time using the
    Hilbert Transform on a 3D array representing signals.
    
    :param T: It seems like you were about to provide some information about the parameter `T` for the
    `hilbert_transform_analysis` function. Could you please provide more details or specify what `T`
    represents in this context?
    :return: The `hilbert_transform_analysis` function returns the mean amplitude and mean phase
    calculated from the Hilbert transform analysis of the input 3D array `T`.
    """
    height, width, frames = T.shape

    # Initialize arrays
    amplitude = np.zeros((height, width, frames))
    phase = np.zeros((height, width, frames))

    # Apply Hilbert Transform
    for i in range(height):
        for j in range(width):
            signal = T[i, j, :]
            analytic_signal = hilbert(signal)
            amplitude[i, j, :] = np.abs(analytic_signal)
            phase[i, j, :] = np.unwrap(np.angle(analytic_signal))

    # Optionally, compute mean amplitude and phase over time
    mean_amplitude = np.mean(amplitude, axis=2)
    mean_phase = np.mean(phase, axis=2)

    return mean_amplitude, mean_phase


def thermal_signal_reconstruction(T, order=5):
    """
    The function `thermal_signal_reconstruction` reconstructs thermal signals using polynomial fitting
    on the logarithm of the input signal.
    
    :param T: It seems like you were about to provide some information about the parameter `T` in the
    `thermal_signal_reconstruction` function. Could you please provide more details or specify the shape
    and type of the `T` parameter so that I can assist you further with the function?
    :param order: The `order` parameter in the `thermal_signal_reconstruction` function represents the
    degree of the polynomial used for fitting the signal data. A higher order polynomial can capture
    more complex patterns in the data but may also be more prone to overfitting. You can adjust the
    `order` parameter to control, defaults to 5 (optional)
    :return: The function `thermal_signal_reconstruction` returns the reconstructed thermal signal array
    `T_reconstructed`.
    """
    height, width, frames = T.shape
    log_time = np.log(np.arange(1, frames + 1))

    # Initialize reconstructed signal array
    T_reconstructed = np.zeros_like(T)

    for i in range(height):
        for j in range(width):
            signal = T[i, j, :]
            log_signal = np.log(signal + np.finfo(float).eps)  # Avoid log(0)
            # Polynomial fitting
            coeffs = np.polyfit(log_time, log_signal, order)
            # Reconstruct signal
            log_signal_fit = np.polyval(coeffs, log_time)
            T_reconstructed[i, j, :] = np.exp(log_signal_fit)

    return T_reconstructed


def modulated_thermography(T, fs, f_stim, harmonics=[2, 3]):
    """
    The function `modulated_thermography` performs demodulation on a thermal image sequence using
    modulated thermography technique to extract amplitude and phase information for specified harmonics.
    
    :param T: It seems like you were about to provide some information about the parameter T in the
    `modulated_thermography` function. Could you please specify what T represents in this context?
    :param fs: The parameter `fs` in the `modulated_thermography` function represents the sampling
    frequency of the thermal data `T`. It is used to calculate the time vector `t` based on the number
    of frames in the data. If you have any more questions or need further clarification, feel
    :param f_stim: The `f_stim` parameter in the `modulated_thermography` function represents the
    frequency of the stimulus signal used in modulated thermography. This frequency is used to generate
    reference signals for demodulation at different harmonics
    :param harmonics: The `harmonics` parameter in the `modulated_thermography` function is a list that
    specifies the harmonics for which you want to analyze the modulated thermography data. By default,
    the function calculates the amplitude and phase for the 2nd and 3rd harmonics
    :return: The function `modulated_thermography` returns two dictionaries: `amplitude` and `phase`.
    The `amplitude` dictionary contains the calculated amplitudes for each harmonic specified in the
    `harmonics` list, while the `phase` dictionary contains the calculated phases for each harmonic.
    """
    height, width, frames = T.shape
    t = np.arange(frames) / fs

    # Initialize dictionaries to hold amplitude and phase for each harmonic
    amplitude = {}
    phase = {}

    for h in harmonics:
        # Reference signals for each harmonic
        ref_sin = np.sin(2 * np.pi * f_stim * h * t)
        ref_cos = np.cos(2 * np.pi * f_stim * h * t)

        I = np.zeros((height, width))
        Q = np.zeros((height, width))

        # Demodulate
        for i in range(height):
            for j in range(width):
                signal = T[i, j, :]
                I[i, j] = (2 / frames) * np.sum(signal * ref_cos)
                Q[i, j] = (2 / frames) * np.sum(signal * ref_sin)

        # Store amplitude and phase
        amplitude[h] = np.sqrt(I**2 + Q**2)
        phase[h] = np.arctan2(Q, I)

    return amplitude, phase
