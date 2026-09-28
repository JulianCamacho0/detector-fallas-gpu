from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "telemetria_publica.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo.joblib"

TARGET_COLUMN = "estado"
ID_COLUMNS = ["episodio_id", "segundo"]
RAW_FEATURE_COLUMNS = ["temp_c", "power_w", "util_pct", "clock_mhz", "ecc_errors"]

# Tamano de la ventana (segundos) usado para agregar telemetria cruda en features.
WINDOW_SIZE = 30

RANDOM_STATE = 42
