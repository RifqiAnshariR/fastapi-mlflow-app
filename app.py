import sys
from pathlib import Path
from enum import Enum
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Body
import mlflow
from mlflow import MlflowClient
from utils.mlflow_setup import MLFlowSetup
from utils.config import config
from utils.logger import default_logger as logger
from run_pipeline import run_pipeline


src_path = Path(__file__) / "src"
sys.path.insert(0, str(src_path))


mlflow_config_values = config.get('mlflow_config', {})
mlflow_config = {'registered_model_prefix': mlflow_config_values.get('registered_model_prefix', None),}


REGISTERED_MODEL_PREFIX = mlflow_config['registered_model_prefix']


app = FastAPI()


class MetricEnum(str, Enum):
    r2 = "r2"
    mse = "mse"
    mae = "mae"
    rmse = "rmse"


class ModelEnum(str, Enum):
    linear_regression = f"{REGISTERED_MODEL_PREFIX}_linear_regression"
    ridge_regression = f"{REGISTERED_MODEL_PREFIX}_ridge_regression"
    lasso_regression = f"{REGISTERED_MODEL_PREFIX}_lasso_regression"
    elastic_net = f"{REGISTERED_MODEL_PREFIX}_elastic_net"
    random_forest = f"{REGISTERED_MODEL_PREFIX}_random_forest"
    gradient_boosting = f"{REGISTERED_MODEL_PREFIX}_gradient_boosting"


class StageEnum(str, Enum):
    Production = "Production"
    Staging = "Staging"
    NoneStage = "None"


@app.on_event("startup")
def startup_event():
    app.state.experiment = MLFlowSetup().setup_mlflow()
    logger.info("MLflow setup complete on app startup.")


@app.get("/ping")
def ping():
    return {"message": "Hello!"}


@app.post("/pipeline/run")
def run_full_pipeline():
    try:
        success = run_pipeline()
        if not success:
            msg = f"Error run full pipeline."
            logger.error(msg)
            raise HTTPException(status_code=500, 
                                detail={
                                    "success": False, 
                                    "data": None, 
                                    "error": msg
                                })

        return {"message": "Pipeline runs successfully"}

    except Exception as e:
        logger.error(f"Error run full pipeline: {str(e)}")
        raise HTTPException(status_code=500, 
                            detail={
                                "success": False, 
                                "data": None, 
                                "error": str(e)
                            })


@app.get("/mlflow/runs")
def get_all_runs():
    try:
        df_info = mlflow.search_runs(experiment_ids=[app.state.experiment.experiment_id])

        if df_info.empty:
            msg = "No runs found."
            logger.warning(msg)
            return {
                "success": True, 
                "data": None, 
                "error": msg
            }

        df_info_clean = df_info.replace({np.nan: None, np.inf: None, -np.inf: None})
        all_runs = df_info_clean.to_dict(orient="records")

        logger.info(f"All runs info retrieved.")

        return {
            "success": True, 
            "data": {
                "experiment": vars(app.state.experiment),
                "all_runs": all_runs
            },
            "error": None
        }

    except Exception as e:
        logger.error(f"Error retrieving MLflow runs: {str(e)}")
        raise HTTPException(status_code=500, 
                            detail={
                                "success": False, 
                                "data": None, 
                                "error": str(e)
                            })


@app.get("/mlflow/best_model")
def get_best_model(metric: MetricEnum = Query(..., description="Metric to rank models.")):
    try:
        order_by = "ASC" if metric in ["mse", "rmse"] else "DESC"
        order_by_args = f"metrics.{metric} {order_by}"

        df_info = mlflow.search_runs(experiment_ids=[app.state.experiment.experiment_id], order_by=[order_by_args])

        if df_info.empty:
            msg = "No runs found."
            logger.warning(msg)
            return {
                "success": True, 
                "data": None, 
                "error": msg
            }

        df_info_clean = df_info.replace({np.nan: None, np.inf: None, -np.inf: None})
        best_row = df_info_clean.iloc[0]
        best_row_data = best_row.to_dict()

        logger.info(f"Best model run info retrieved.")

        return {
            "success": True,
            "data": {
                "experiment": vars(app.state.experiment),
                "best_run": best_row_data
            },
            "error": None
        }

    except Exception as e:
        logger.error(f"Error in get_best_model: {str(e)}")
        raise HTTPException(status_code=500, 
                            detail={
                                "success": False, 
                                "data": None, 
                                "error": str(e)
                            })


@app.get("/mlflow/status")
def get_model_status(run_id: str = Query(..., description="Run id of model to check the status.")):     # From model registry
    try:
        versions = mlflow.search_model_versions(filter_string=f"run_id='{run_id}'")
        if versions is None:
            msg = f"Versions of model with run_id: '{run_id}' not found."
            logger.error(msg)
            raise HTTPException(status_code=404, 
                                detail={
                                    "success": False, 
                                    "data": None, 
                                    "error": msg
                                })

        return {
            "success": True, 
            "data": [vars(v) for v in versions],
            "error": None
        }

    except Exception as e:
        logger.error(f"Error in check_model_versions: {str(e)}")
        raise HTTPException(status_code=500, 
                            detail={
                                "success": False, 
                                "data": None, 
                                "error": str(e)
                            })


@app.post("/mlflow/transition/")
def transition_model(
    registered_model_name: ModelEnum = Query(..., description="Registered model name."),
    model_version: int = Query(..., description="Target version."),
    to_stage: StageEnum = Query(..., description="Target stage.")
):
    try:
        client = MlflowClient()

        client.transition_model_version_stage(
            name=registered_model_name.value,
            version=model_version,
            stage=to_stage.value
        )

        return {
            "success": True,
            "data": {
                "registered_model_name": registered_model_name.value,
                "model_version": model_version,
                "to_stage": to_stage.value
            },
            "error": None
        }

    except Exception as e:
        logger.error(f"Error in transitioning model stage: {str(e)}")
        raise HTTPException(status_code=500, 
                            detail={
                                "success": False, 
                                "data": None, 
                                "error": str(e)
                            })



@app.post("/mlflow/inference/")
def predict_production(
    model_name: ModelEnum = Query(..., description="Select model for inference"),
    input_data: dict = Body(..., description="Input features for prediction")
):
    try:
        model = mlflow.pyfunc.load_model(f"models:/{model_name.value}/Production")
        input_df = pd.DataFrame([input_data])  # single row df.

        predictions = model.predict(input_df)

        return {
            "success": True,
            "data": predictions.tolist(),
            "error": None
        }

    except Exception as e:
        logger.error(f"Error during production inference: {str(e)}")
        raise HTTPException(status_code=500, 
                            detail={
                                "success": False,
                                "data": None,
                                "error": str(e)
                            })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
