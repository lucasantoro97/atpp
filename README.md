
# ATPP - Active Thermography Test Post-Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/github/actions/workflow/status/lucasantoro97/atpp/docs.yml?branch=main)](https://github.com/lucasantoro97/atpp/actions/workflows/docs.yml)

## Overview

ATPP is a Python library for post-processing FLIR thermal camera data.

Active Thermography Test Post-Processing (ATPP) is a powerful tool designed to help users analyze and process thermography data. This application offers a range of post-processing techniques, including data filtering, noise reduction, and advanced visualization options for enhanced analysis.

- **Author**: Luca Santoro
- **Version**: 
- **License**: MIT

## Features

- Automated thermography data processing.
- Noise reduction filters.
- Advanced visualization tools (e.g., thermal maps, time-lapse).
- Seamless integration with various thermography file formats.
- Easy-to-use API for custom data handling.

For detailed documentation on features, please visit the [ATPP Documentation](https://lucasantoro97.github.io/atpp/).

## Requirements

Before installation, ensure that the following dependencies are met:

- Python 3.x
- Required libraries (listed in `requirements.txt`)

## Installation

To install ATPP, follow these steps:

```bash
pip install atpp
```

To install directly from PyPI, use:

```bash
pip install atpp
```

For more detailed installation instructions, refer to the [Installation Guide](https://lucasantoro97.github.io/atpp/).

## Usage

Once installed, ATPP can be used with the following commands:

```bash

```python
from atpp import FlirVideo

# Example usage of ATPP library
flir_video = FlirVideo('path_to_file.ats')
flir_video.process_data()

```

For a detailed guide on how to use ATPP, visit the [Usage Documentation](https://lucasantoro97.github.io/atpp/).

## Contributing

We welcome contributions from the community. Please read the [Contributing Guidelines](https://lucasantoro97.github.io/atpp/) before submitting any pull requests.

## Support

If you encounter any issues or need support, feel free to check the [FAQ](https://lucasantoro97.github.io/atpp/) or open an issue in our [GitHub Issue Tracker](https://github.com/lucasantoro97/atpp/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Special thanks to everyone who has contributed to this project, including the open-source community, whose work made this project possible.