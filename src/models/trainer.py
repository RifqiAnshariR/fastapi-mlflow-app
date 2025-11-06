import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from pathlib import Path
from utils.logger import default_logger as logger
import pickle


class ModelTrainer:    
    def __init__(self, 
                 X_train, X_val, y_train, y_val, 
                 preprocessor, 
                 trained_model_path, trained_feature_importance_path):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.preprocessor = preprocessor
        self.trained_model_path = Path(trained_model_path)
        self.trained_feature_importance_path = Path(trained_feature_importance_path)
        logger.info(f"Initialized ModelTrainer.")


    def _prepare_trained_path(self):
        self.trained_model_path.mkdir(parents=True, exist_ok=True)
        self.trained_feature_importance_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Trained model path prepared at {self.trained_model_path} \
            and trained feature importance path prepared at {self.trained_feature_importance_path}")


    def _evaluate_model(self, y_val, y_pred):
        try:
            metrics = {
                'r2': r2_score(y_val, y_pred),
                'mae': mean_absolute_error(y_val, y_pred),
                'mse': mean_squared_error(y_val, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_val, y_pred))
            }
            return metrics

        except Exception as e:
            logger.error(f"Error calculating metrics: {str(e)}")
            raise


    def train_and_evaluate_model(self, model_name, model_params, model_instance):
        try:
            self._prepare_trained_path()

            # if mlflow.active_run():
            #     mlflow.end_run()

            with mlflow.start_run(run_name=model_name):
                model = Pipeline(steps=[
                    ('preprocessor', self.preprocessor.preprocessor),
                    ('regressor', model_instance)
                ])

                model.fit(self.X_train, self.y_train)

                # Save model to local
                model_filename = f"{model_name}.pkl"
                model_full_path = str(self.trained_model_path / model_filename)
                with open(model_full_path, "wb") as file:
                    pickle.dump(model, file)

                mlflow.sklearn.log_model(model, 
                                         model_name, 
                                         registered_model_name=f"house_price_prediction_{model_name}")
                mlflow.log_artifact(local_path=model_full_path,
                                    artifact_path="trained_model")
                logger.info(f"Trained model: {model_filename} saved to: {self.trained_model_path}")

                y_pred = model.predict(self.X_val)
                metrics = self._evaluate_model(self.y_val, y_pred)

                mlflow.log_metrics(metrics)
                mlflow.log_params(model_params)
                logger.info(f"Trained model: {model_name} with metrics: {metrics}")

                regressor = model.named_steps['regressor']
                if hasattr(regressor, "coef_"):
                    importance_values = regressor.coef_
                elif hasattr(regressor, "feature_importances_"):
                    importance_values = regressor.feature_importances_
                else:
                    importance_values = None

                if importance_values is not None:
                    feature_importance = pd.DataFrame({
                        'feature': model.named_steps['preprocessor'].get_feature_names_out(),
                        'coefficients/importance': importance_values
                    })

                    # Save feature importance to local
                    feature_importance_filename = f"feature_importance_{model_name}.csv"
                    feature_importance_full_path = str(self.trained_feature_importance_path / feature_importance_filename)
                    feature_importance.to_csv(path_or_buf=feature_importance_full_path, 
                                              index=False)

                    mlflow.log_artifact(local_path=feature_importance_full_path, 
                                        artifact_path="trained_feature_importance")
                    logger.info(f"Feature importance: {feature_importance_filename} saved to: {self.trained_feature_importance_path}")
                    
                mlflow.log_artifact("requirements.txt", artifact_path="dependencies")

        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            raise
