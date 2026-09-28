"""Preparación de features/target y definición del pipeline de preprocesamiento."""

import pandas as pd

from src.features import build_window_features, FEATURE_NAMES
from src.config import  TARGET_COLUMN
   

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara las features a partir del DataFrame crudo validado y construido por ventana."""
    return build_window_features(df)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Construye las features y separa el DataFrame en variables predictoras (X) y variable objetivo (y)."""
    df = build_features(df)

    X = df[FEATURE_NAMES]
    y = df[TARGET_COLUMN]
    return X, y


