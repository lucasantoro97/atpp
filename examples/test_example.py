# The `TestExample` class contains unit tests for the `select_folder_and_process` and `process_video`
# functions in the `example` module, using mock objects to isolate dependencies.
import unittest
"""
Unit tests for the example module.
This module contains unit tests for the functions `select_folder_and_process` and `process_video`
from the `example` module. The tests use the `unittest` framework and mock various dependencies
to isolate the functionality being tested.
Classes:
    TestExample: Contains unit tests for the `select_folder_and_process` and `process_video` functions.
Methods:
    test_select_folder_and_process(self, mock_process_video, mock_FlirVideo, mock_listdir, mock_makedirs, mock_askdirectory):
        Tests the `select_folder_and_process` function by mocking dependencies such as file dialogs,
        directory creation, file listing, and video processing.
    test_process_video(self, mock_lock_in_amplifier):
        Tests the `process_video` function by mocking the `lock_in_amplifier` function and verifying
        that it is called with the correct arguments. Also checks if the function prints 'ok!'.
"""
from unittest.mock import patch, MagicMock
import os
import sys
from example import select_folder_and_process, process_video

# Add the path to the example.py file
sys.path.append('/home/luca/Documents/atpp/examples')


class TestExample(unittest.TestCase):
    """
    TestExample is a test case class for testing the functionality of the example module.
    Methods:
        test_select_folder_and_process(mock_process_video, mock_FlirVideo, mock_listdir, mock_makedirs, mock_askdirectory):
            Tests the select_folder_and_process function by mocking dependencies and verifying the expected behavior.
        test_process_video(mock_lock_in_amplifier):
            Tests the process_video function by mocking dependencies and verifying the expected behavior.
    """

    @patch('example.filedialog.askdirectory')
    @patch('example.os.makedirs')
    @patch('example.os.listdir')
    @patch('example.FlirVideo')
    @patch('example.process_video')
    def test_select_folder_and_process(self, mock_process_video, mock_FlirVideo, mock_listdir, mock_makedirs, mock_askdirectory):
        # Mock the folder selection
        mock_askdirectory.return_value = '/mock/folder'
        
        # Mock the list of files in the directory
        mock_listdir.return_value = ['file1.ats', 'file2.txt', 'file3.ats']
        
        # Call the function
        select_folder_and_process()
        
        # Check if the result directory was created
        mock_makedirs.assert_any_call('/mock/folder/result', exist_ok=True)
        
        # Check if the subdirectories for .ats files were created
        mock_makedirs.assert_any_call('/mock/folder/result/file1', exist_ok=True)
        mock_makedirs.assert_any_call('/mock/folder/result/file3', exist_ok=True)
        
        # Check if FlirVideo was called with the correct file paths
        mock_FlirVideo.assert_any_call('/mock/folder/file1.ats')
        mock_FlirVideo.assert_any_call('/mock/folder/file3.ats')
        
        # Check if process_video was called with the correct arguments
        self.assertEqual(mock_process_video.call_count, 2)
    
    @patch('example.lim.lock_in_amplifier')
    def test_process_video(self, mock_lock_in_amplifier):
        """
        Test the process_video function.
        This test verifies the following:
        1. A mock FlirVideo object is created.
        2. The lock_in_amplifier function is mocked to return a specific value.
        3. The process_video function is called with the mock FlirVideo object and a mock results path.
        4. The lock_in_amplifier function is called with the correct arguments.
        5. The process_video function prints 'ok!'.
        Args:
            mock_lock_in_amplifier (MagicMock): Mocked lock_in_amplifier function.
        """
        # Create a mock FlirVideo object
        mock_flir_video = MagicMock()
        
        # Mock the lock_in_amplifier function
        mock_lock_in_amplifier.return_value = (1.0, 0.5)
        
        # Call the function
        process_video(mock_flir_video, '/mock/results')
        
        # Check if lock_in_amplifier was called with the correct arguments
        mock_lock_in_amplifier.assert_called_once_with(mock_flir_video, frequency=1)
        
        # Check if the function prints 'ok!'
        with patch('builtins.print') as mock_print:
            process_video(mock_flir_video, '/mock/results')
            mock_print.assert_called_with('ok!')

if __name__ == '__main__':
    unittest.main()