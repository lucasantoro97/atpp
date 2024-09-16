# ATPP - Active Thermography Test Post-Processing

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Documentation Status](https://img.shields.io/github/actions/workflow/status/lucasantoro97/atpp/deploy_docs.yml?branch=main)](https://lucasantoro97.github.io/atpp/)
[![GitHub Pages](https://img.shields.io/badge/docs-GitHub%20Pages-blue.svg)](https://lucasantoro97.github.io/atpp/)


## Overview

Active Thermography Test Post-Processing (ATPP) is a powerful tool designed to help users analyze and process thermography data. This application offers a range of post-processing techniques, including data filtering, noise reduction, and advanced visualization options for enhanced analysis.

- **Author**: [Luca Santoro](https://lucasantoro97.github.io/cv/)
- **Version**: 0.1
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

## Installation

To install ATPP, follow these steps:

```bash
git clone https://github.com/lucasantoro97/atpp.git
cd atpp
pip install 
```

To install directly from PyPI (coming soon), use:

```bash
pip install atpp
```

For more detailed installation instructions, refer to the [Installation Guide](https://lucasantoro97.github.io/atpp/).

## Usage

Once installed, ATPP can be used with the following commands:

```bash
Usage Guide
===========

This section explains how to use ATPP after installation. Below are some
typical usage examples and commands.

Basic Usage
-----------

Once ATPP is installed, you can use the following commands:

``` {.bash}
atpp process <input_data>
```

This will run the default processing pipeline on the provided
thermography data.

Advanced Options
----------------

You can pass in various flags for more control:

``` {.bash}
atpp process <input_data> --filter noise_reduction --visualize true
```

This will apply noise reduction filtering and enable visualization for
the thermography data.

For a full list of commands and options, use:

``` {.bash}
atpp --help
```

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