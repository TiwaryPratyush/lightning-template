from functools import wraps
from src.utils.logging_utils import setup_logging

logger = setup_logging()

def train_eval_wrapper(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            logger.info(f"Starting {func.__name__}")
            result = func(*args, **kwargs)
            logger.info(f"Finished {func.__name__}")
            return result
        except Exception as e:
            logger.exception(f"An error occurred in {func.__name__}: {str(e)}")
            raise
    return wrapper