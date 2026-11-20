"""
ModelWatch - Model Eğitim ve MLflow Kayıt Modülü
Referans veri üzerinde RandomForestClassifier eğitir, baseline verilerini saklar ve MLflow Registry'ye kaydeder.
"""

import json
import os
import tempfile
from pathlib import Path

import mlflow
from mlflow.client import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from src.data_loader import generate_training_data


def train_and_register_model(
    model_name: str = "ModelWatch-CustomerSatisfaction",
    alias: str = "staging",
    tracking_uri: str | None = None,
    n_samples: int = 5000,
    n_estimators: int = 150,
    max_depth: int | None = 10,
    random_state: int = 42,
    target_col: str = "memnun",
) -> dict:
    """Eğitim verisiyle model eğitir, baseline metriklerini ve referans veriyi loglar ve alias atar."""
    resolved_uri = tracking_uri or os.getenv(
        "MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"
    )
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment("modelwatch_customer_satisfaction")

    df = generate_training_data(n_samples=n_samples, random_state=random_state)
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    model_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "random_state": random_state,
        "n_jobs": -1,
    }
    logged_params = {
        "n_estimators": n_estimators,
        "max_depth": max_depth if max_depth is not None else -1,
        "n_samples": n_samples,
        "random_state": random_state,
        "model_type": "RandomForestClassifier",
    }

    clf = RandomForestClassifier(**model_params)

    with mlflow.start_run(run_name="rf_baseline_train") as run:
        clf.fit(X_train, y_train)
        val_preds = clf.predict(X_val)

        baseline_pos_rate = round(float(y.mean()), 4)
        metrics = {
            "val_accuracy": round(float(accuracy_score(y_val, val_preds)), 4),
            "val_f1": round(float(f1_score(y_val, val_preds, zero_division=0)), 4),
            "val_precision": round(
                float(precision_score(y_val, val_preds, zero_division=0)), 4
            ),
            "val_recall": round(
                float(recall_score(y_val, val_preds, zero_division=0)), 4
            ),
            "baseline_positive_rate": baseline_pos_rate,
        }
        mlflow.log_params(logged_params)
        mlflow.log_metrics(metrics)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            ref_csv = tmp_path / "reference_data.csv"
            df.to_csv(ref_csv, index=False)
            mlflow.log_artifact(str(ref_csv), artifact_path="reference")

            baseline_json = tmp_path / "baseline_metrics.json"
            baseline_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(baseline_json), artifact_path="reference")

        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            name="model",
            registered_model_name=model_name,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        client = MlflowClient(tracking_uri=resolved_uri)
        client.set_registered_model_alias(
            name=model_name,
            alias=alias,
            version=str(model_info.registered_model_version),
        )

        return {
            "run_id": run.info.run_id,
            "model_name": model_name,
            "version": str(model_info.registered_model_version),
            "alias": alias,
            "val_metrics": metrics,
        }


if __name__ == "__main__":
    result = train_and_register_model()
    print("Registered model:", result)
