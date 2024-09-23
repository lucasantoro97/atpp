import atpp 
from atpp.fnv_class import FlirVideo
from atpp import lock_in_imaging as lim
from atpp import advanced_imaging as aim
import os
import numpy as np
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt
from atpp.plt_style import set_plt_style



# Apply the custom style
set_plt_style()

from atpp.plt_style import get_custom_cmap
import time as ttime
custom_cmap = get_custom_cmap()


#parameters
LI_freq = 1
RESAMPLING_RATIO   =   10 #resampling ratio for the data (10 means 10 times less of the lock-in frequency)

def select_folder_and_process():
    """
    Prompts the user to select a folder, processes all '.ats' files in the selected folder,
    and saves the results in a 'result' directory within the selected folder.

    The function performs the following steps:
    1. Opens a file dialog for the user to select a folder.
    2. If no folder is selected, prints a message and exits.
    3. Creates a 'result' directory within the selected folder.
    4. Iterates over all files in the selected folder.
    5. For each file with a '.ats' extension:
       a. Creates a subdirectory within the 'result' directory named after the file (without extension).
       b. Initializes a FlirVideo object with the file path.
       c. Processes the video file and saves the results in the corresponding subdirectory.

    Note:
        - This function requires the `tkinter` and `os` modules.
        - The `FlirVideo` class and `process_video` function must be defined elsewhere in the code.

    Returns:
        None
    """
    root = tk.Tk()
    
    root.withdraw()
    folder_selected = filedialog.askdirectory()

    if not folder_selected:
        print("No folder selected")
        return

    results_dir = os.path.join(folder_selected, 'result')
    os.makedirs(results_dir, exist_ok=True)

    for file_name in os.listdir(folder_selected):
        if file_name.endswith('.ats'):
            file_path = os.path.join(folder_selected, file_name)
            file_results_dir = os.path.join(results_dir, os.path.splitext(file_name)[0])
            os.makedirs(file_results_dir, exist_ok=True)
            # get the FlirVideo object
            flir_video = FlirVideo(file_path)
            # process the file
            process_video(flir_video, file_results_dir)
    
    
    # folder_selected = 'path/to/folder' #change this to the path of the folder you want to process

    # results_dir = os.path.join(folder_selected, 'result')
    # os.makedirs(results_dir, exist_ok=True)

    # for file_name in os.listdir(folder_selected):
    #     if file_name.endswith('.ats'):
    #         file_path = os.path.join(folder_selected, file_name)
    #         file_results_dir = os.path.join(results_dir, os.path.splitext(file_name)[0])
    #         os.makedirs(file_results_dir, exist_ok=True)
    #         # get the FlirVideo object
    #         flir_video = FlirVideo(file_path)
    #         # process the file
    #         process_video(flir_video, file_results_dir)

def process_video(flir_video, results_dir):
    #do stuff with the flir_video object
    
    T=flir_video.temperature
    fs=flir_video.framerate
    print('fs:', fs)
    time=flir_video.time
    # plt.plot(time, np.max(np.max(T, axis=0), axis=0))
    # plt.show()
    #desample
    T,time,fs = lim.desample(T, time , fs, LI_freq, RESAMPLING_RATIO)
    
    print('New framrate after desampling:', fs)
    
    start_frame, end_frame = lim.find_se_frames(T, fs=fs)
    # cut the data in the range
    T = T[:, :, start_frame:end_frame]
    time = time[start_frame:end_frame]
    
    # do fft of T signal and plot frequency spectrum
    # Perform FFT on the maximum temperature signal
    T_max = np.max(np.max(T, axis=0), axis=0)
    n = len(T_max)
    T_max_fft = np.fft.fft(T_max)
    frequencies = np.fft.fftfreq(n, d=1/fs)

    # Only take the positive frequencies
    positive_frequencies = frequencies[:n // 2]
    positive_T_max_fft = np.abs(T_max_fft[:n // 2])

    # Plot the amplitude of the FFT
    # plt.figure()
    # plt.plot(positive_frequencies, 20 * np.log10(positive_T_max_fft))
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude (dB)')
    # plt.title('Amplitude of T_max vs Frequency (dB Scale)')
    # # plt.grid(True)
    # plt.show()
    
    
    

    
    f_stim = LI_freq

    # Measure time for phase coherence imaging
    start_time = ttime.time()
    amplitude_map, phase_map, phase_coherence_map = aim.phase_coherence_imaging(T, fs, f_stim)
    phase_coherence_time = ttime.time() - start_time

    # Measure time for synchronous demodulation
    start_time = ttime.time()
    demodulated_amplitude, demodulated_phase = aim.synchronous_demodulation(T, fs, f_stim)
    synchronous_demodulation_time = ttime.time() - start_time

    # Measure time for Hilbert transform analysis
    start_time = ttime.time()
    hil_amplitude, hil_phase = aim.hilbert_transform_analysis(T)
    hilbert_transform_time = ttime.time() - start_time
    #calculate standard deviation of phase and amplitude along time axis for each pixel
    # Only consider the last 10 seconds of data
    # last_10_seconds = time >= (time[-1] - 10)
    # hil_phase_last_10 = hil_phase[:, :, last_10_seconds]
    # hil_amplitude_last_10 = hil_amplitude[:, :, last_10_seconds]

    # phase_std = np.std(hil_phase_last_10, axis=2)
    # amplitude_std = np.std(hil_amplitude_last_10, axis=2)
    # print(hil_phase)
    phase_std = np.std(hil_phase, axis=2)
    amplitude_std = np.std(hil_amplitude, axis=2)
    
    
    
    # # Plot the standard deviation of amplitude vs time for each pixel
    # plt.figure()
    # for i in range(0, amplitude_std.shape[0], 500):
    #     for j in range(0, amplitude_std.shape[1], 500):
    #         plt.plot(time, hil_amplitude[i, j, :], alpha=0.5)
    # plt.xlabel('Time')
    # plt.ylabel('Amplitude std')
    # plt.title('Amplitude std vs Time for each pixel')
    # plt.show()

    # # Plot the standard deviation of phase vs time for each pixel
    # plt.figure()
    # for i in range(0, phase_std.shape[0], 500):
    #     for j in range(0, phase_std.shape[1], 500):
    #         plt.plot(time, hil_phase[i, j, :], alpha=0.5)
    # plt.xlabel('Time')
    # plt.ylabel('Phase std')
    # plt.title('Phase std vs Time for each pixel')
    # plt.show()

    
    #rename as mean amplitude and phase for lazy reasons
    mean_amplitude = amplitude_std
    mean_phase = phase_std

    # Measure time for modulated thermography
    start_time = ttime.time()
    modulated_amplitude_map, modulated_phase_map = aim.modulated_thermography(T, fs, f_stim, harmonics=[3]) #just third harmonic
    modulated_thermography_time = ttime.time() - start_time
    #transform modulated amplitude map and phase from dictionary to np.array
    modulated_amplitude_map = modulated_amplitude_map[3]
    modulated_phase_map = modulated_phase_map[3]
    

    # Mask data
    mask = lim.mask_data(amplitude_map, threshold=0.5)
    amplitude_map = np.where(mask, amplitude_map, np.nan)
    phase_map = np.where(mask, phase_map, np.nan)
    phase_coherence_map = np.where(mask, phase_coherence_map, np.nan)
    demodulated_amplitude = np.where(mask, demodulated_amplitude, np.nan)
    demodulated_phase = np.where(mask, demodulated_phase, np.nan)
    mean_amplitude = np.where(mask, mean_amplitude, np.nan)
    mean_phase = np.where(mask, mean_phase, np.nan)
    modulated_amplitude_map = np.where(mask, modulated_amplitude_map, np.nan)
    modulated_phase_map = np.where(mask, modulated_phase_map, np.nan)

    # Visualize the results
    plt.figure(figsize=(15, 15))

    # Find the bounding box of the masked data
    mask_indices = np.argwhere(mask)
    (y_start, x_start), (y_end, x_end) = mask_indices.min(0), mask_indices.max(0) + 1

    # Calculate percentiles to remove outliers
    vmin_amp, vmax_amp = np.percentile(amplitude_map[mask], [2, 98])
    vmin_phase, vmax_phase = np.percentile(phase_map[mask], [2, 98])
    vmin_phase_coh, vmax_phase_coh = np.percentile(phase_coherence_map[mask], [2, 98])
    vmin_demod_amp, vmax_demod_amp = np.percentile(demodulated_amplitude[mask], [2, 98])
    vmin_demod_phase, vmax_demod_phase = np.percentile(demodulated_phase[mask], [2, 98])
    vmin_mean_amp, vmax_mean_amp = np.percentile(mean_amplitude[mask], [2, 98])
    vmin_mean_phase, vmax_mean_phase = np.percentile(mean_phase[mask], [2, 98])
    vmin_mod_amp, vmax_mod_amp = np.percentile(modulated_amplitude_map[mask], [2, 98])
    vmin_mod_phase, vmax_mod_phase = np.percentile(modulated_phase_map[mask], [2, 98])

    plt.subplot(3, 3, 1)
    plt.imshow(amplitude_map, cmap='gray', vmin=vmin_amp, vmax=vmax_amp)
    plt.title('Amplitude Map')
    plt.colorbar()
    plt.xlim(x_start, x_end)
    plt.ylim(y_end, y_start)  # Note: y-axis is inverted in images

    plt.subplot(3, 3, 2)
    plt.imshow(phase_map, cmap='gray', vmin=vmin_phase, vmax=vmax_phase)
    plt.title('Phase Map')
    plt.colorbar()
    plt.xlim(x_start, x_end)
    plt.ylim(y_end, y_start)  # Note: y-axis is inverted in images

    plt.subplot(3, 3, 3)
    plt.imshow(phase_coherence_map, cmap='gray', vmin=vmin_phase_coh, vmax=vmax_phase_coh)
    plt.title('Phase Coherence Map')
    plt.colorbar()
    plt.xlim(x_start, x_end)
    plt.ylim(y_end, y_start)  # Note: y-axis is inverted in images

    plt.subplot(3, 3, 4)
    plt.imshow(demodulated_amplitude, cmap='gray', vmin=vmin_demod_amp, vmax=vmax_demod_amp)
    plt.title('Demodulated Amplitude')
    plt.colorbar()
    plt.xlim(x_start, x_end)
    plt.ylim(y_end, y_start)  # Note: y-axis is inverted in images

    plt.subplot(3, 3, 5)
    plt.imshow(demodulated_phase, cmap='gray', vmin=vmin_demod_phase, vmax=vmax_demod_phase)
    plt.title('Demodulated Phase')
    plt.colorbar()
    plt.xlim(x_start, x_end)
    plt.ylim(y_end, y_start)  # Note: y-axis is inverted in images

    plt.subplot(3, 3, 6)
    plt.imshow(mean_amplitude, cmap='gray', vmin=vmin_mean_amp, vmax=vmax_mean_amp)
    plt.title('Mean Amplitude')
    plt.colorbar()
    plt.xlim(x_start, x_end)
    plt.ylim(y_end, y_start)  # Note: y-axis is inverted in images

    plt.subplot(3, 3, 7)
    plt.imshow(mean_phase, cmap='gray', vmin=vmin_mean_phase, vmax=vmax_mean_phase)
    plt.title('Mean Phase')
    plt.colorbar()
    plt.xlim(x_start, x_end)
    plt.ylim(y_end, y_start)  # Note: y-axis is inverted in images

    plt.subplot(3, 3, 8)
    plt.imshow(modulated_amplitude_map, cmap='gray', vmin=vmin_mod_amp, vmax=vmax_mod_amp)
    plt.title('Modulated Amplitude Map')
    plt.colorbar()
    plt.xlim(x_start, x_end)
    plt.ylim(y_end, y_start)  # Note: y-axis is inverted in images

    plt.subplot(3, 3, 9)
    plt.imshow(modulated_phase_map, cmap='gray', vmin=vmin_mod_phase, vmax=vmax_mod_phase)
    plt.title('Modulated Phase Map')
    plt.colorbar()
    plt.xlim(x_start, x_end)
    plt.ylim(y_end, y_start)  # Note: y-axis is inverted in images

    plt.tight_layout()
    plt.show()

    # Print and save the profiling results
    profiling_results = (
        f"Phase Coherence Imaging Time: {phase_coherence_time:.4f} seconds\n"
        f"Synchronous Demodulation Time: {synchronous_demodulation_time:.4f} seconds\n"
        f"Hilbert Transform Analysis Time: {hilbert_transform_time:.4f} seconds\n"
        f"Modulated Thermography Time: {modulated_thermography_time:.4f} seconds\n"
    )

    print(profiling_results)

    with open("profiling_results.txt", "w") as file:
        file.write(profiling_results)


select_folder_and_process()


