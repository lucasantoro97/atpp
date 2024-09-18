Installation Guide
==================

This section provides instructions on how to install ATPP (Active Thermography Test Post-Processing) from the GitHub repository.

System Requirements
-------------------

Before installing ATPP, make sure your system meets the following requirements:

- **Python**: Version 3.x or higher
- **pip**: Python package manager (ensure it is up to date)
- **fnv** library from `Flir Science File SDK <https://www.flir.it/products/flir-science-file-sdk/?vertical=rd%20science&segment=solutions>`_

Installing ATPP from GitHub
---------------------------

To install ATPP from the GitHub source, clone the repository and install the package manually.

Steps:

1. **Clone the repository** from GitHub:

   .. code-block:: bash

      git clone https://github.com/lucasantoro97/atpp.git

2. **Navigate to the project directory**:

   .. code-block:: bash

      cd atpp

3. **Install the package using pip**:

   .. code-block:: bash

      pip install .

Setting Up a Virtual Environment (Optional)
-------------------------------------------

It is recommended to use a virtual environment to manage dependencies. Here's how you can set up a virtual environment:

.. code-block:: bash

    # Create a virtual environment
    python3 -m venv venv

    # Activate the virtual environment
    source venv/bin/activate  # On macOS/Linux
    .\venv\Scripts\activate  # On Windows

    # Install ATPP in the virtual environment
    pip install .

Verifying the Installation
--------------------------

To verify that ATPP has been installed correctly, you can check the installed package version:

.. code-block:: bash

    atpp --version

Troubleshooting
---------------

If you encounter issues during installation, ensure that you have the latest version of Python and pip. You can upgrade pip by running:

.. code-block:: bash

    pip install --upgrade pip

For more information on specific errors, visit the `ATPP Issue Tracker <https://github.com/lucasantoro97/atpp/issues>`_ or open a new issue if necessary.
