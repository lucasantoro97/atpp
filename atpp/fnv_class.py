import os
import fnv
import fnv.file
import fnv.reduce
import numpy as np


class FlirVideo:
    """
    The `FlirVideo` class initializes a Flir video object with specified parameters, loads data from the
    video file, and calculates the framerate based on the number of frames and time duration.
    
    :param file_name: The `file_name` parameter is a string that represents the name of the file
    containing the FLIR video data that you want to process
    :param emissivity: The `emissivity` parameter is a float that represents the emissivity value to be set.
    :param reflected_temp: The `reflected_temp` parameter is a float that represents the reflected temperature to be set.
    """
    def __init__(self, file_name, emissivity=None, reflected_temp=None):
        self.file_name = file_name
        self.imager = fnv.file.ImagerFile(file_name)
        self.emissivity = emissivity
        self.reflected_temp = reflected_temp
        self._set_object_parameters()
        self._load_data()
        self.framerate = self.imager.num_frames / self.time[-1]
        print("File:", self.file_name)
        print("Framerate:", self.framerate)
        print("Time:", self.time[-1])
        print("Number of frames:", self.imager.num_frames)  

    def _set_object_parameters(self):
        obj_params = self.imager.object_parameters
        obj_params.emissivity = self.emissivity if self.emissivity is not None else 0.9
        obj_params.reflected_temp = self.reflected_temp if self.reflected_temp is not None else 293.15
        self.imager.object_parameters = obj_params

    def _load_data(self):
        self.temperature = np.empty([self.imager.height, self.imager.width, self.imager.num_frames])
        self.time = np.empty(self.imager.num_frames)
        self.imager.get_frame(0)
        start_time = self.imager.frame_info.time

        if self.imager.has_unit(fnv.Unit.TEMPERATURE_FACTORY):
            self.imager.unit = fnv.Unit.TEMPERATURE_FACTORY
            self.imager.temp_type = fnv.TempType.CELSIUS
        else:
            self.imager.unit = fnv.Unit.COUNTS

        for i in range(self.imager.num_frames):
            self.imager.get_frame(i)
            self.time[i] = (self.imager.frame_info.time - start_time).total_seconds()
            self.temperature[..., i] = np.array(self.imager.final, copy=False).reshape((self.imager.height, self.imager.width))
