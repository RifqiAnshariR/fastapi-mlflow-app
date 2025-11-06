import pandas as pd
from pathlib import Path
from utils.logger import default_logger as logger


class DataLoader:
    def __init__(self, data_path, irrelevant_features, categorical_features, numerical_features, target):
        self.data_path = Path(data_path)
        self.irrelevant_features = irrelevant_features
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.target = target
        logger.info(f"Initialized DataLoader.")


    def load_data(self):
        try:
            df = pd.read_csv(self.data_path)
            logger.info(f"Data loaded from: {self.data_path} successfully with shape: {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise


    def validate_data(self, df):
        try:
            required_columns = [self.irrelevant_features] + self.categorical_features + self.numerical_features + [self.target]
            missing_columns = [col for col in required_columns if col not in df.columns]

            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return False

            null_counts = df.isna().sum()
            if null_counts.any():
                logger.warning(f"Found null values: {null_counts[null_counts > 0].to_dict()}")

            duplicate_counts = df.duplicated().sum()
            if duplicate_counts:
                logger.warning(f"Found duplicate values: {duplicate_counts}")

            logger.info("Data validation completed")
            return True

        except Exception as e:
            logger.error(f"Error validating data: {e}")
            return False
