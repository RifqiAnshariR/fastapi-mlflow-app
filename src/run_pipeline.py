import os
import sys
from pathlib import Path
import mlflow
from sklearn.model_selection import train_test_split
from data.loader import DataLoader
from data.cleaner import DataCleaner
from data.preprocessor import DataPreprocessor
from models.configurator import ModelFactory
from models.trainer import ModelTrainer
from utils.logger import default_logger as logger
from utils.config import config


src_path = str(Path(__file__).parent.parent)
sys.path.append(src_path)


config_values = {
    'train_dataset_path': config.get('train_dataset_path', "data/train.csv"),
    'test_dataset_path': config.get('test_dataset_path', "data/test.csv"),
    'trained_model_path': config.get('trained_model_path', "artifacts/trained_model_path"),
    'trained_feature_importance_path': config.get('trained_feature_importance_path', "artifacts/trained_feature_importance_path"),
    'model_config': config.get('model_config', {}),
    'dataset_columns': config.get('dataset_columns', {}),
    'mlflow_config': config.get('mlflow_config', {}),
}
model_config_values = config_values['model_config']
model_config = {
    'linear_regression': model_config_values.get('linear_regression', []),
    'ridge_regression': model_config_values.get('ridge_regression', []),
    'lasso_regression': model_config_values.get('lasso_regression', []),
    'elastic_net': model_config_values.get('elastic_net', []),
    'random_forest': model_config_values.get('random_forest', []),
    'gradient_boosting': model_config_values.get('gradient_boosting', []),
}
dataset_config_values = config_values['dataset_columns']
dataset_config = {
    'irrelevant_features': dataset_config_values.get('irrelevant_features', None),
    'categorical_features': dataset_config_values.get('categorical_features', []),
    'numerical_features': dataset_config_values.get('numerical_features', []),
    'target': dataset_config_values.get('target', None),
}
mlflow_config_values = config_values['mlflow_config']
mlflow_config = {
    'registered_model_prefix': mlflow_config_values.get('registered_model_prefix', None)
}


TRAIN_DATASET_PATH = config_values['train_dataset_path']    # Main dataset.
TEST_DATASET_PATH = config_values['test_dataset_path']
TRAINED_MODEL_PATH = config_values['trained_model_path']
TRAINED_FEATURE_IMPORTANCE_PATH = config_values['trained_feature_importance_path']
MODEL_CONFIG = config_values['model_config']
IRRELEVANT_FEATURES = dataset_config['irrelevant_features']
CATEGORICAL_FEATURES = dataset_config['categorical_features']
NUMERICAL_FEATURES = dataset_config['numerical_features']
TARGET = dataset_config['target']
REGISTERED_MODEL_PREFIX = mlflow_config['registered_model_prefix']


def run_pipeline():
    try:
        logger.separator()
        logger.info("Starting pipeline execution")

        # 1. Load Data
        logger.info("Step 1: Loading data")
        data_loader = DataLoader(data_path=TRAIN_DATASET_PATH,
                                 irrelevant_features=IRRELEVANT_FEATURES, 
                                 categorical_features=CATEGORICAL_FEATURES, 
                                 numerical_features=NUMERICAL_FEATURES,
                                 target=TARGET)

        df = data_loader.load_data()

        if not data_loader.validate_data(df):
            logger.warning("Unable to validate data.")
            raise

        # 2. Cleaning data
        logger.info("Step 2: Cleaning data")
        data_cleaner = DataCleaner(df=df, 
                                   irrelevant_features=IRRELEVANT_FEATURES, 
                                   categorical_features=CATEGORICAL_FEATURES, 
                                   numerical_features=NUMERICAL_FEATURES, 
                                   target=TARGET)

        X, y = data_cleaner.clean()

        # 3. Setup Preprocessor
        logger.info("Step 3: Setup Preprocessor")
        preprocessor = DataPreprocessor(numerical_cols=NUMERICAL_FEATURES,
                                        categorical_cols=CATEGORICAL_FEATURES)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Setup Configurator and Train Model
        logger.info("Step 4: Setup Configurator and Train Model")
        for model_name, model_params in MODEL_CONFIG.items():
            model_instance = ModelFactory().create_model(model_name=model_name,
                                                         model_params=model_params)

            model_trainer = ModelTrainer(X_train=X_train, 
                                            X_val=X_val, 
                                            y_train=y_train, 
                                            y_val=y_val,
                                            preprocessor=preprocessor, 
                                            trained_model_path=TRAINED_MODEL_PATH, 
                                            trained_feature_importance_path=TRAINED_FEATURE_IMPORTANCE_PATH)

            model_trainer.train_and_evaluate_model(model_name=model_name,
                                                   model_params=model_params,
                                                   model_instance=model_instance)

        logger.info("Pipeline execution completed successfully")
        return True

    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        return False
