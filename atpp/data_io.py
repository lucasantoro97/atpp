import os
import fnv
import fnv.file
import fnv.reduce
import numpy as np
from atpp.logging_config import logger
import tempfile
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import psutil
import platform

# Move init_worker and process_frame to the module level
def init_worker(init_args):
    global worker_imager, worker_start_time
    file_name, emissivity, reflected_temp, unit, temp_type = init_args

    worker_imager = fnv.file.ImagerFile(file_name)
    obj_params = worker_imager.object_parameters
    obj_params.emissivity = emissivity if emissivity is not None else 0.9
    obj_params.reflected_temp = reflected_temp if reflected_temp is not None else 293.15
    worker_imager.object_parameters = obj_params

    if worker_imager.has_unit(fnv.Unit.TEMPERATURE_FACTORY):
        worker_imager.unit = unit
        worker_imager.temp_type = temp_type
    else:
        worker_imager.unit = fnv.Unit.COUNTS

    worker_imager.get_frame(0)
    worker_start_time = worker_imager.frame_info.time

def process_frame(args):
    i, height, width = args
    global worker_imager, worker_start_time
    worker_imager.get_frame(i)
    time_i = (worker_imager.frame_info.time - worker_start_time).total_seconds()
    temperature_i = np.array(worker_imager.final, copy=False).reshape((height, width))
    return i, time_i, temperature_i

def load_flir_video(file_name, MEMMAP=None, emissivity=None, reflected_temp=None):
    """
    Loads data from the FLIR video file and calculates the temperature, time, and framerate.

    :param file_name: The `file_name` parameter is a string that represents the name of the file
    containing the FLIR video data that you want to process.
    :param memmap_flag: Flag indicating whether to use memory mapping to disk.
    :param emissivity: The `emissivity` parameter is a float that represents the emissivity value to be set.
    :param reflected_temp: The `reflected_temp` parameter is a float that represents the reflected temperature to be set.
    :return: Tuple containing the temperature array, time array, and framerate.
    """
    
    #Get file size
    file_size = os.path.getsize(file_name)
    #Get available RAM
    if platform.system() == 'Windows':
        available_RAM = psutil.virtual_memory().available
    else:
        available_RAM = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')

    memmap_flag = available_RAM/10 <= file_size if MEMMAP is None else MEMMAP
    
    # Print available RAM and file size ratio
    available_RAM_MB = available_RAM / (1024 ** 2)
    file_size_MB = file_size / (1024 ** 2)
    logger.info(f"Available RAM: {available_RAM_MB:.2f} MB, File size: {file_size_MB:.2f} MB, Ratio: {available_RAM_MB / file_size_MB:.2f}")

    if memmap_flag:
        logger.info("Importing data on disk...")
    else:
        logger.info("Importing data on RAM....")

    # Open the file to get metadata
    imager = fnv.file.ImagerFile(file_name)
    obj_params = imager.object_parameters
    obj_params.emissivity = emissivity if emissivity is not None else 0.9
    obj_params.reflected_temp = reflected_temp if reflected_temp is not None else 293.15
    imager.object_parameters = obj_params

    # Get start time and metadata
    imager.get_frame(0)
    start_time = imager.frame_info.time
    height = imager.height
    width = imager.width
    num_frames = imager.num_frames

    # Set unit and temperature type
    if imager.has_unit(fnv.Unit.TEMPERATURE_FACTORY):
        unit = fnv.Unit.TEMPERATURE_FACTORY
        temp_type = fnv.TempType.CELSIUS
    else:
        unit = fnv.Unit.COUNTS
        temp_type = None

    # Preallocate arrays
    if memmap_flag:
        temp_dir = tempfile.gettempdir()
        temp_file_path = os.path.join(temp_dir, f"{os.path.basename(file_name)}.dat")
        temperature = np.memmap(temp_file_path, dtype=np.float32, mode='w+', shape=(height, width, num_frames))
        logger.info("Data imported on disk")
    else:
        temperature = np.empty((height, width, num_frames), dtype=np.float32)

    time = np.empty(num_frames, dtype=np.float64)

    # Prepare arguments for initializer
    init_args = (file_name, emissivity, reflected_temp, unit, temp_type)

    # Prepare arguments for process_frame
    args_list = [(i, height, width) for i in range(num_frames)]

    # Use multiprocessing Pool to process frames in parallel
    with Pool(processes=cpu_count(), initializer=init_worker, initargs=(init_args,)) as pool:
        results = list(tqdm(
            pool.imap_unordered(process_frame, args_list),
            total=num_frames,
            desc="Processing frames"
        ))

    # Collect results
    for i, time_i, temperature_i in results:
        time[i] = time_i
        temperature[:, :, i] = temperature_i

    # Ensure memmap data is flushed to disk
    if memmap_flag:
        temperature.flush()

    framerate = num_frames / time[-1] if time[-1] != 0 else 0

    return temperature, time, framerate
