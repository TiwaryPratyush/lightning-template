import os
from loguru import logger

def setup_logging():
    log_file = os.path.join("logs", "app.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    logger.add(log_file, rotation="10 MB", level="INFO")
    return logger