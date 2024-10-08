# ATPP - Active Thermography Test Post-Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/github/actions/workflow/status/lucasantoro97/atpp/deploy_docs.yml?branch=main)](https://lucasantoro97.github.io/atpp/)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://lucasantoro97.github.io/atpp/)
![GitHub Issues](https://img.shields.io/github/issues/lucasantoro97/atpp)

## Overview

Active Thermography Test Post-Processing (ATPP) is a powerful tool designed to help users analyze and process thermography data. This application offers a range of post-processing techniques, including data filtering, noise reduction, and advanced visualization options for enhanced analysis.

- **Author**: [Luca Santoro](https://lucasantoro97.github.io/cv/)
- **Version**: placeholder
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
- fnv library from [Flir Science File SDK](https://www.flir.it/products/flir-science-file-sdk/?vertical=rd%20science&segment=solutions)

## Build Status

| Python Version | Build Status |
| -------------- | ------------ |
| 3.9            | ![Build Status](https://github.com/lucasantoro97/atpp/actions/workflows/python-package.yml/badge.svg?branch=main&event=push&matrix.python-version=3.9) |
| 3.10           | ![Build Status](https://github.com/lucasantoro97/atpp/actions/workflows/python-package.yml/badge.svg?branch=main&event=push&matrix.python-version=3.10) |
| 3.11           | ![Build Status](https://github.com/lucasantoro97/atpp/actions/workflows/python-package.yml/badge.svg?branch=main&event=push&matrix.python-version=3.11) |

## Installation

To install ATPP, follow these steps:

```bash
git clone https://github.com/lucasantoro97/atpp.git
cd atpp
pip install .
```


For more detailed installation instructions, refer to the [Installation Guide](https://lucasantoro97.github.io/atpp/).

## Usage

Once installed, ATPP can be used with the following commands to see the first frame of the video:

```bash
import atpp

atpp.process('<input flir video>',visualize=True)
```

For a detailed guide on how to use ATPP, visit the [Usage Documentation](https://lucasantoro97.github.io/atpp/).


## Contributing

We welcome contributions from the community. Please read the [Contributing Guidelines](https://lucasantoro97.github.io/atpp/) before submitting any pull requests.

## Support

If you encounter any issues or need support, feel free to check the [FAQ](https://lucasantoro97.github.io/atpp/) or open an issue in our [GitHub Issue Tracker](https://github.com/lucasantoro97/atpp/issues).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use ATPP in your research, please consider citing it as follows:

### BibTeX Citation
```bibtex
@misc{santoro2024atpp,
    author = {Luca Santoro},
    title = {ATPP - Active Thermography Test Post-Processing},
    year = {2024},
    howpublished = {\url{https://github.com/lucasantoro97/atpp}},
    note = {Accessed: September 2024}
}
```
### Plain Text Citation
Santoro, L. (2024). ATPP - Active Thermography Test Post-Processing. Available at: https://github.com/lucasantoro97/atpp (Accessed: September 2024).

## Acknowledgments

Special thanks to everyone who has contributed to this project, including the open-source community, whose work made this project possible.