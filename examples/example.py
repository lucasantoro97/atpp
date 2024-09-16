import atpp 
from atpp.fnv_class import FlirVideo
from atpp import lock_in_imaging as lim
import os
import tkinter as tk


from tkinter import filedialog




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

def process_video(flir_video, results_dir):
    #do stuff with the flir_video object
    amplitude, phase = lim.lock_in_amplifier(flir_video, frequency=1)
    print('ok!')
    return


select_folder_and_process()

