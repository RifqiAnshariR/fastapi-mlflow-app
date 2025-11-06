from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from utils.logger import default_logger as logger


class ModelFactory:    
    @classmethod
    def create_model(cls, model_name, model_params):
        models = {
            'linear_regression': LinearRegression,
            'ridge_regression': Ridge,
            'lasso_regression': Lasso,
            'elastic_net': ElasticNet,
            'random_forest': RandomForestRegressor,
            'gradient_boosting': GradientBoostingRegressor
        }

        try:
            logger.info(f"Creating regression model: {model_name}")
            
            if model_name not in models:
                logger.warning(f"Unknown model type: {model_name}")
                raise ValueError(f"Unknown model type: {model_name}")

            model_class = models[model_name]
            model_instance = model_class(**model_params)
            logger.info(f"Successfully created: {model_name} model")
            return model_instance

        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise
