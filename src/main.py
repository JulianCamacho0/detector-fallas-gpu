"""Punto de entrada del pipeline de Machine Learning: python src/main.py"""

import sys
from pathlib import Path

# Permite ejecutar este archivo directamente (python src/main.py) sin instalar el paquete.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import load_data
from src.schema import validate_data

def main() -> None:
    print("Iniciando pipeline de entrenamiento...")

    df = load_data()
    print(f"Datos crudos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")

    df = validate_data(df)
    print("Validación del esquema completada correctamente")

if __name__ == "__main__":
    main()