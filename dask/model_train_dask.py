
import argparse
import logging
import platform
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import dask.dataframe as dd
import pandas as pd
import xgboost as xgb
import yaml
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--models-dir", required=True)
    return parser.parse_args()

def get_model_instance(name, params):
    model_map = {
        "LinearRegression": LinearRegression,
        "RandomForest": RandomForestRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "XGBoost": xgb.XGBRegressor
    }
    return model_map[name](**params)

def main(args):
    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]

    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    mlruns_path = models_dir / "mlruns"
    mlflow.set_tracking_uri(f"file://{mlruns_path.resolve()}")
    mlflow.set_experiment(model_cfg.get("name", "default-experiment"))

    # Load dataset lazily with Dask
    df = dd.read_csv(args.data).compute()
    target = model_cfg["target_variable"]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=model_cfg.get("test_size", 0.2), random_state=42
    )

    model = get_model_instance(model_cfg["best_model"], model_cfg.get("parameters", {}))
    save_path = models_dir / model_cfg.get("artifact_name", "trained_model.pkl")

    with mlflow.start_run(run_name="final_training") as run:
        run_id = run.info.run_id
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        mlflow.log_params(model_cfg.get("parameters", {}))
        mlflow.log_metrics({"mae": mae, "r2": r2})
        mlflow.sklearn.log_model(model, "tuned_model")
        joblib.dump(model, save_path)

        metrics_path = models_dir / "metrics.json"
        metrics_path.write_text(f'{{"mae": {mae}, "r2": {r2}}}')

    logger.info(f"Model saved at: {save_path}")
    logger.info(f"Metrics saved at: {metrics_path}")

if __name__ == "__main__":
    args = parse_args()
    main(args)
