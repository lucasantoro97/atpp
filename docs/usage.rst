
Usage Guide
===========

This section explains how to use ATPP after installation. Below are some typical usage examples and commands.

Basic Usage
-----------

Once ATPP is installed, you can use the following commands:

.. code-block:: bash

   import atpp

   atpp.process('<input flir video>')

This will run the default processing pipeline on the provided thermography data.

Advanced Options
----------------

You can pass in various flags for more control:

.. code-block:: bash

   atpp.process <input_data> --filter noise_reduction --visualize true

This will apply noise reduction filtering and enable visualization for the thermography data.

For a full list of commands and options, use:

.. code-block:: bash

   atpp --help
