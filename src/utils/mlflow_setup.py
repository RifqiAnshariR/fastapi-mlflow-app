import mlflow
from utils.logger import default_logger as logger
from utils.config import config


mlflow_config = config.get('mlflow_config', {})
TRACKING_URI = mlflow_config.get('tracking_uri', 'sqlite:///mlflow.db')
EXPERIMENT_NAME = mlflow_config.get('experiment_name', 'house_price_prediction')


class MLFlowSetup:    
    def __init__(self):
        self.tracking_uri=TRACKING_URI
        self.experiment_name=EXPERIMENT_NAME
        

    def setup_mlflow(self):
        try:
            mlflow.set_tracking_uri(self.tracking_uri)
            experiment = mlflow.set_experiment(self.experiment_name)
            logger.info(f"Initialized MLFlowSetup with experiment: {experiment.experiment_id}")

            return experiment

        except Exception as e:
            logger.error(f"Error initialize up MLflow: {str(e)}")
            raise
