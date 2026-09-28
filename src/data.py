"""Lectura de los datos crudos del proyecto."""

from pathlib import Path

import pandas as pd

from src.config import DATA_PATH


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    """Carga el dataset de tickets desde `path` y lo devuelve como DataFrame."""
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de datos en: {path}")
    return pd.read_csv(path)