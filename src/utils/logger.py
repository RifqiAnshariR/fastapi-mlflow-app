import logging
import sys
from pathlib import Path
import logging.handlers


LOG_NAME = "House Price Prediction"
LOG_DIR = "./logs/house_price_prediction.log"


class CustomLogger:    
    @staticmethod
    def setup_logger(log_name, log_file):
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s | %(name)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.TimedRotatingFileHandler(
                log_file,
                when='midnight',
                interval=1,
                backupCount=7
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
        def separator(char="=", length=100):
                line = char * length
                logger.info(line)

        logger.separator = separator

        return logger


default_logger = CustomLogger.setup_logger(log_name=LOG_NAME, 
                                           log_file=LOG_DIR)
