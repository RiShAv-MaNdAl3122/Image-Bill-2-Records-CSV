import os
import logging

def setup_logger() -> logging.Logger:
    """
    Sets up logging to both console and a log file in the logs/ directory.
    """
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger("ImageBillExtractor")
    logger.setLevel(logging.INFO)
    
    # Avoid adding multiple handlers if logger is already configured
    if not logger.handlers:
        formatter = logging.Formatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler
        file_handler = logging.FileHandler(os.path.join(log_dir, "processing.log"), encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

logger = setup_logger()
