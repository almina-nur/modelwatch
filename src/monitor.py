"""
ModelWatch - Model İzleme (Model Monitoring) Modülü
Kayıtlı MLflow modelini yükleyerek üretim verisindeki tahmin ve performans metriklerini izler.
"""
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import mlflow


def evaluate_model_monitoring(
    df: pd.DataFrame,
    model_name: str = "ModelWatch-CustomerSatisfaction",
    alias: str = "staging",
    tracking_uri: str = "sqlite:///mlflow.db",
    target_col: str = "memnun",
    run_tag: str = "current",
) -> dict:
    """
    Verilen veri seti üzerinde modeli çalıştırır; tahmin dağılımı ve
    hedef sütun ('memnun') mevcutsa sınıflandırma metriklerini döndürür.
    """
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{model_name}@{alias}"

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]

    model = mlflow.pyfunc.load_model(model_uri)
    preds = model.predict(X)
    preds_arr = np.asarray(preds).ravel()

    result = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_tag": run_tag,
        "model_uri": model_uri,
        "n_samples": len(df),
        "positive_rate": round(float(np.mean(preds_arr == 1)), 4),
        "prediction_counts": {
            "0": int(np.sum(preds_arr == 0)),
            "1": int(np.sum(preds_arr == 1)),
        },
        "metrics": None,
    }

    if target_col in df.columns:
        y_true = df[target_col].to_numpy().ravel()
        result["metrics"] = {
            "accuracy": round(float(accuracy_score(y_true, preds_arr)), 4),
            "f1": round(float(f1_score(y_true, preds_arr, zero_division=0)), 4),
            "precision": round(float(precision_score(y_true, preds_arr, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, preds_arr, zero_division=0)), 4),
        }

    return result
