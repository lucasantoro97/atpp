"""_summary_
This module configures the logging for the application.

It sets up logging to output messages to both the terminal and a log file.
The logging level is set to INFO by default, but can be changed to DEBUG for more detailed logs.

Logging Configuration:
- Level: INFO (default), can be changed to DEBUG
- Format: '%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
- Handlers:
    - StreamHandler: Outputs logs to the terminal (stdout)
    - FileHandler: Outputs logs to 'ThernessLibCore.log' file
- Force: True (forces reconfiguration of logging)

Logger:
- Name: 'therness'
"""
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Change to DEBUG for more detailed logs
    format='%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),  # Ensure logs are shown in terminal
        logging.FileHandler("ThernessLibCore.log"),  # Log to a file
    ],
    force=True  # Force reconfiguration of logging
)

# Create a logger
logger = logging.getLogger('therness')
