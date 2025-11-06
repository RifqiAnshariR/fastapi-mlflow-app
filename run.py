import sys
from pathlib import Path

src_path = Path(__file__) / "src"
sys.path.insert(0, str(src_path))

from utils.mlflow_setup import MLFlowSetup
from run_pipeline import run_pipeline

if __name__ == "__main__":
    MLFlowSetup().setup_mlflow()

    success = run_pipeline()
    
    if success:
        print("Pipeline completed successfully")
    else:
        print("Pipeline failed")
