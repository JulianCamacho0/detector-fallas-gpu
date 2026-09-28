"""Punto de entrada del pipeline de Machine Learning: python src/main.py"""

import sys
from pathlib import Path

# Permite ejecutar este archivo directamente (python src/main.py) sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TEST_SIZE
from src.data import load_data
from src.schema import validate_data
from src.preprocessing import split_features_target

def main() -> None:
    print("Iniciando pipeline de entrenamiento...")

    df = load_data()
    print(f"Datos crudos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    df = validate_data(df)
    print("Validación del esquema completada correctamente")

    X, y = split_features_target(df)
    print(f"Features y target separados: X={X.shape[0]} filas, {X.shape[1]} columnas; y={y.shape[0]} filas")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} filas | Test: {len(X_test)} filas")

if __name__ == "__main__":
    main()