###  README.md

# ATPP - Active Thermography Test Post-Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Introduction

ATPP is a Python library for post-processing FLIR thermal camera data.

## Installation

```bash
pip install atpp
Usage

```python
from atpp import FlirVideo

# Example usage of ATPP library
flir_video = FlirVideo('path_to_file.ats')
flir_video.process_data()


License
This project is licensed under the MIT License.