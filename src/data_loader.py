"""ModelWatch - Veri Yukleme ve Simulasyon Modulu."""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

SPLIT_SEED = 42

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


def _base_population(n_total: int = 10000, seed: int = SPLIT_SEED) -> pd.DataFrame:
    X, y = make_classification(
        n_samples=n_total,
        n_features=8,
        n_informative=6,
        n_redundant=1,
        n_classes=2,
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["memnun"] = y
    return df


def generate_training_data(
    n_samples: int = 5000, random_state: int = 42
) -> pd.DataFrame:
    """Referans / egitim veri setini uretim havuzundan ayri bolum olarak uretir."""
    pop = _base_population(n_total=10000, seed=SPLIT_SEED)
    train_pool, _ = train_test_split(
        pop, test_size=0.4, random_state=SPLIT_SEED, stratify=pop["memnun"]
    )
    rng = np.random.default_rng(random_state)
    replace = n_samples > len(train_pool)
    idx = rng.choice(len(train_pool), size=n_samples, replace=replace)
    return train_pool.iloc[idx].reset_index(drop=True).copy()


def generate_production_data(
    n_samples: int = 1000, drift: bool = False, random_state: int = 101
) -> pd.DataFrame:
    """Egitim bolumunden bagimsiz ayri havuzdan uretim veri seti uretir."""
    pop = _base_population(n_total=10000, seed=SPLIT_SEED)
    _, prod_pool = train_test_split(
        pop, test_size=0.4, random_state=SPLIT_SEED, stratify=pop["memnun"]
    )
    rng = np.random.default_rng(random_state)
    replace = n_samples > len(prod_pool)
    idx = rng.choice(len(prod_pool), size=n_samples, replace=replace)
    df = prod_pool.iloc[idx].reset_index(drop=True).copy()
    if drift:
        df["gelir"] = df["gelir"] + 1.5 + rng.normal(0.0, 0.15, size=n_samples)
        df["aylik_ucret"] = (
            df["aylik_ucret"] + 1.5 + rng.normal(0.0, 0.15, size=n_samples)
        )
    return df
