"""
Logging configuration for the pipeline.
"""
import logging
import sys
from pathlib import Path
from datetime import datetime
import colorlog

from config.settings import LOGS_DIR, DEBUG, VERBOSE


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Set up logger with colored console output and file logging.
    
    Args:
        name: Logger name
        log_file: Optional log file path
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Set level based on config
    if DEBUG:
        logger.setLevel(logging.DEBUG)
    elif VERBOSE:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARNING)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler with colors
    console_handler = colorlog.StreamHandler(sys.stdout)
    console_formatter = colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'green',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white',
        }
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = LOGS_DIR / f'{name}_{datetime.now().strftime("%Y%m%d")}.log'
    
    file_handler = logging.FileHandler(log_file)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    return logger


# Create default loggers
pipeline_logger = setup_logger('pipeline', LOGS_DIR / 'pipeline.log')
error_logger = setup_logger('errors', LOGS_DIR / 'errors.log')
