"""Preparación de features/target y definición del pipeline de preprocesamiento."""

import pandas as pd

from src.features import build_window_features
from src.config import  TARGET_COLUMN
   

def _prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepara las features a partir del DataFrame crudo validado y construido por ventana."""
    return build_window_features(df)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Construye las features y separa el DataFrame en variables predictoras (X) y variable objetivo (y)."""
    df = _prepare_features(df)

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


