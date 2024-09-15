###  README.md

```markdown
# ATPP - Active Thermography Test Post-Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Introduction

ATPP (Active Thermography Test Post-Processing) is a Python library designed to post-process FLIR thermal camera data for active thermography tests. It supports lock-in amplifier processing, pulsed diffusivity calculation, and custom plotting styles for efficient and accurate analysis.

This library is designed to work with `.ats` files generated by FLIR cameras, and it provides utilities to calculate the amplitude, phase, and thermal diffusivity for thermography analysis.

### Features

- **FLIR Thermal Data Processing**: Efficient handling and processing of `.ats` files.
- **Lock-in Amplifier Analysis**: Extraction of amplitude and phase information using the lock-in amplifier technique.
- **Pulsed Diffusivity Calculation**: Robust tools for calculating thermal diffusivity from pulsed thermography data.
- **Custom Plotting Styles**: Consistent plotting styles for generating scientific figures.
- **File Explorer Integration**: Easy folder selection for batch processing of multiple `.ats` files.

## Installation

To install the package from GitHub and the required dependencies, clone the repository and install the package:

```bash
git clone https://github.com/yourusername/atpp.git
cd atpp
pip install .
```

Alternatively, you can install the dependencies from `requirements.txt` and then install the package:

```bash
pip install -r requirements.txt
pip install .
```

## Usage

Here’s an example of how to use the library to process a folder containing FLIR `.ats` files and plot the results.

### Example: Basic File Processing

```python
from atpp import FlirVideo, lock_in_amplifier
import os
from tkinter import filedialog, Tk

def select_folder_and_process():
    root = Tk()
    root.withdraw()
    folder_selected = filedialog.askdirectory()

    if not folder_selected:
        print("No folder selected")
        return

    results_dir = os.path.join(folder_selected, 'results')
    os.makedirs(results_dir, exist_ok=True)

    for file_name in os.listdir(folder_selected):
        if file_name.endswith('.ats'):
            file_path = os.path.join(folder_selected, file_name)
            file_results_dir = os.path.join(results_dir, os.path.splitext(file_name)[0])
            os.makedirs(file_results_dir, exist_ok=True)

            # Initialize the FLIR video object
            flir_video = FlirVideo(file_path)
            
            # Perform lock-in analysis
            amplitude, phase = lock_in_amplifier(flir_video, frequency=1)
            print("Amplitude and Phase Calculated")

select_folder_and_process()
```

### Example: Custom Plotting Style

```python
import matplotlib.pyplot as plt
from atpp import set_plt_style, get_custom_cmap

# Apply the custom style
set_plt_style()

# Generate some data and use the custom colormap
custom_cmap = get_custom_cmap()
data = np.random.rand(10, 10)
plt.imshow(data, cmap=custom_cmap)
plt.colorbar()
plt.show()
```

### Example: Pulsed Diffusivity Calculation

```python
from atpp import plot_line_through_centroid, calculate_diffusivity

# Assuming you have a FLIR video object
flir_video = FlirVideo('path_to_ats_file.ats')

# Process the thermal data
sigma_values, sigma_times = plot_line_through_centroid(flir_video, num_frames=50, resolution=0.2, radius=5)

# Calculate the diffusivity
diffusivity, slope, intercept, sigma_values, sigma_times, sigma_square = calculate_diffusivity(sigma_values, sigma_times)
print(f"Calculated Diffusivity: {diffusivity}")
```

## File Structure

- `script/` - Contains the core processing and utility functions.
  - `fnv_class.py` - FLIR video processing class for loading `.ats` files.
  - `lock_in_diffusivity.py` - Lock-in amplifier analysis functions for diffusivity.
  - `lock_in_imaging.py` - Lock-in amplifier imaging functions.
  - `plt_style.py` - Custom matplotlib plotting styles.
  - `pulsed_diffusivity.py` - Pulsed diffusivity calculation tools.
- `examples/` - Example scripts demonstrating file processing and analysis.
  - `example.py` - Basic example for processing `.ats` files.
  - `plottingstyle_example.py` - Example demonstrating the use of custom plotting styles.
- `LICENSE` - Project license (MIT).
- `README.md` - Project documentation.
- `requirements.txt` - Dependencies for the project.
- `setup.py` - Setup script for installing the package.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributions

We welcome contributions! If you'd like to contribute to this project, please open a pull request or issue on GitHub.

