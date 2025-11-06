import yaml
from pathlib import Path
from utils.logger import default_logger as logger


CONFIG_PATH = "config/config.yml"


class Config:
    def __init__(self, config_path):
        self.config_path = Path(config_path)
        self.config = self._load_config()

    
    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)

            logger.info(f"Initialized Config.")
            return config

        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
        

    def get(self, key, default):
        return self.config.get(key, default)


config = Config(config_path=CONFIG_PATH)
