import logging


# Configure logging
def setup_logger():
    """Setup logger with both file and stream handlers"""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # turn off propagation to parent logger
    logger.propagate = False
    
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s',
                                datefmt='%Y-%m-%d %H:%M:%S')
    
    # add file handler
    fh = logging.FileHandler('qtlscan.log')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    # add console handler
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    return logger

# create logger instance
logger = setup_logger()