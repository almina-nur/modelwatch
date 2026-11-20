"""
ModelWatch - Model Eğitim ve MLflow Kayıt Modülü
Referans veri üzerinde RandomForestClassifier eğitir ve MLflow Registry'ye kaydeder.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import mlflow
from mlflow.client import MlflowClient

from src.data_loader import generate_training_data


def train_and_register_model(
    model_name: str = "ModelWatch-CustomerSatisfaction",
    alias: str = "staging",
    tracking_uri: str = "sqlite:///mlflow.db",
    n_samples: int = 5000,
    random_state: int = 42,
    target_col: str = "memnun",
) -> dict:
    """Eğitim verisiyle model eğitir, metrikleri loglar ve alias atar."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("modelwatch_customer_satisfaction")

    df = generate_training_data(n_samples=n_samples, random_state=random_state)
    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=random_state,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="rf_baseline_train") as run:
        clf.fit(X_train, y_train)
        val_preds = clf.predict(X_val)

        metrics = {
            "val_accuracy": round(float(accuracy_score(y_val, val_preds)), 4),
            "val_f1": round(float(f1_score(y_val, val_preds, zero_division=0)), 4),
            "val_precision": round(float(precision_score(y_val, val_preds, zero_division=0)), 4),
            "val_recall": round(float(recall_score(y_val, val_preds, zero_division=0)), 4),
        }
        mlflow.log_params(
            {
                "n_estimators": 150,
                "max_depth": 10,
                "n_samples": n_samples,
                "random_state": random_state,
            }
        )
        mlflow.log_metrics(metrics)

        model_info = mlflow.sklearn.log_model(
            sk_model=clf,
            artifact_path="model",
            registered_model_name=model_name,
            skops_trusted_types=["sklearn.tree._tree.Tree"],
        )

        client = MlflowClient(tracking_uri=tracking_uri)
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
