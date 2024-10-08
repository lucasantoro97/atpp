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
