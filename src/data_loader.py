"""
ModelWatch - Veri Yükleme ve Simülasyon Modülü
Müşteri memnuniyeti tahmini için referans ve üretim verisi üretir.
"""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

FEATURE_NAMES = [
    "yas",
    "gelir",
    "abonelik_suresi",
    "aylik_ucret",
    "destek_talebi_sayisi",
    "son_etkilesim_gun",
    "kullanilan_hizmet",
    "memnuniyet_skoru",
]


def generate_training_data(
    n_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Referans / eğitim veri seti üretir."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        n_classes=2,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["memnun"] = y
    return df


def generate_production_data(
    n_samples: int = 1000,
    drift: bool = False,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Üretim veri seti üretir.
    drift=False olduğunda referans dağılımından alt örnek çeker.
    drift=True olduğunda gelir ve aylık ücret dağılımlarında kayma simüle edilir.
    """
    full_ref = generate_training_data(n_samples=5000, random_state=42)
    rng = np.random.default_rng(random_state)

    replace = n_samples > len(full_ref)
    idx = rng.choice(len(full_ref), size=n_samples, replace=replace)
    df = full_ref.iloc[idx].reset_index(drop=True).copy()

    if drift:
        df["gelir"] = df["gelir"] + 1.5 + rng.normal(0.0, 0.15, size=n_samples)
        df["aylik_ucret"] = df["aylik_ucret"] + 1.5 + rng.normal(0.0, 0.15, size=n_samples)

    return df
