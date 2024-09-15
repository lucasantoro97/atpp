from script.fnv_class import FlirVideo
import script.lock_in_imaging as lim

import tkinter as tk
import os
from tkinter import filedialog




def select_folder_and_process():
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

def process_video(flir_video, results_dir):
    #do stuff with the flir_video object
    amplitude, phase = lim.lock_in_amplifier(flir_video, frequency=1)
    print('ok!')
    return


select_folder_and_process()

