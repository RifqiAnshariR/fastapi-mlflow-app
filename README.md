# Assignment: MLOPS

## Stack:
1. Runtime: Python 3.13.
2. Experiment tracking & model management tool: MLflow.
3. Backend: FastAPI.

## Prerequisites:
1. -

## Setup:
1. `git clone https://github.com/RifqiAnshariR/fastapi-mlflow-app.git`
2. `cd fastapi-mlflow-app`
3. `py -3.10 -m venv .venv` and activate it `.venv\Scripts\activate`
4. `pip install -r requirements.txt`

## How to run:
1. To run api: `python app.py`
2. To run all pipelines (test all pipelines): `python run.py`

## How to run (via Docker):
1. `docker build -t fastapi-mlflow-app .`
2. `docker run -d --name fastapi-mlflow-app_container -p 80:8000 fastapi-mlflow-app`

## Flowchart:
![flowchart](./assets/flowchart.jpg)
