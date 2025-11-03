"""
Logging configuration for the pipeline.
"""
import logging
import sys


def setup_logger(name: str, log_file: str = None) -> logging.Logger:
    """
    Set up logger with console output.
    
    Args:
        name: Logger name
        log_file: Optional log file path
    
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    
    # Set level
    logger.setLevel(logging.INFO)
    
    # Prevent duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(levelname)-8s %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger


# Create default loggers
pipeline_logger = setup_logger('pipeline')
error_logger = setup_logger('errors')