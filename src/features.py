"""Transformacion de telemetria cruda (una fila por segundo) a features por ventana.

Flujo: telemetria cruda -> ventanas de WINDOW_SIZE segundos -> features agregadas.
Una ventana nunca mezcla segundos de dos episodio_id distintos, y su etiqueta se
hereda del `estado` (unico) del episodio al que pertenece. `estado`, `episodio_id`
y `segundo` nunca se usan como valores de entrada del modelo: `episodio_id` solo
sirve para no partir episodios entre train/test, y `estado` solo se usa como y.
"""

import numpy as np
import pandas as pd

from src.config import TARGET_COLUMN, WINDOW_SIZE

# Nombres de las columnas de features que finalmente ve el modelo (sin ids ni target).
FEATURE_NAMES = [
	# temp_c: el nivel y el rango delatan sobrecalentamiento; la std delata inestabilidad.
	"temp_c_mean",
	"temp_c_std",
	"temp_c_min",
	"temp_c_max",
	"temp_c_range",
	"temp_c_trend",
	# power_w: caidas/picos y alta variabilidad delatan falla_alimentacion.
	"power_w_mean",
	"power_w_std",
	"power_w_min",
	"power_w_max",
	"power_w_range",
	"power_w_trend",
	# util_pct: cae cuando el reloj hace throttling por sobrecalentamiento.
	"util_pct_mean",
	"util_pct_std",
	"util_pct_min",
	"util_pct_max",
	# clock_mhz: throttling (sobrecalentamiento) o inestabilidad (falla_alimentacion).
	"clock_mhz_mean",
	"clock_mhz_std",
	"clock_mhz_min",
	"clock_mhz_max",
	"clock_mhz_range",
	"clock_mhz_trend",
	# ecc_errors: la senal mas directa de degradacion_memoria.
	"ecc_errors_sum",
	"ecc_errors_mean",
	"ecc_errors_max",
	"ecc_errors_nonzero_count",
]


def _trend(values: pd.Series) -> float:
	"""Pendiente de una recta ajustada a la ventana; capta si la senal sube o baja."""
	if len(values) < 2:
		return 0.0
	x = np.arange(len(values), dtype=float)
	slope, _ = np.polyfit(x, values.to_numpy(dtype=float), 1)
	return float(slope)


def _compute_window_features(window: pd.DataFrame) -> dict:
	"""Calcula el vector de features de UNA ventana cruda (filas ordenadas por segundo).

	Reutilizable por el entrenamiento (ventanas de episodios historicos) y, mas
	adelante, por la API (una unica ventana recibida en una peticion).
	"""
	temp = window["temp_c"]
	power = window["power_w"]
	util = window["util_pct"]
	clock = window["clock_mhz"]
	ecc = window["ecc_errors"]

	return {
		"temp_c_mean": temp.mean(),
		"temp_c_std": temp.std(ddof=0),
		"temp_c_min": temp.min(),
		"temp_c_max": temp.max(),
		"temp_c_range": temp.max() - temp.min(),
		"temp_c_trend": _trend(temp),
		"power_w_mean": power.mean(),
		"power_w_std": power.std(ddof=0),
		"power_w_min": power.min(),
		"power_w_max": power.max(),
		"power_w_range": power.max() - power.min(),
		"power_w_trend": _trend(power),
		"util_pct_mean": util.mean(),
		"util_pct_std": util.std(ddof=0),
		"util_pct_min": util.min(),
		"util_pct_max": util.max(),
		"clock_mhz_mean": clock.mean(),
		"clock_mhz_std": clock.std(ddof=0),
		"clock_mhz_min": clock.min(),
		"clock_mhz_max": clock.max(),
		"clock_mhz_range": clock.max() - clock.min(),
		"clock_mhz_trend": _trend(clock),
		"ecc_errors_sum": ecc.sum(),
		"ecc_errors_mean": ecc.mean(),
		"ecc_errors_max": ecc.max(),
		"ecc_errors_nonzero_count": int((ecc > 0).sum()),
	}


def build_window_features(dataframe: pd.DataFrame) -> pd.DataFrame:
	"""Convierte telemetria cruda y validada en un dataframe de features por ventana.

	Cada ventana agrupa hasta WINDOW_SIZE segundos consecutivos de un mismo
	episodio_id (nunca mezcla episodios). El resultado conserva `episodio_id`
	(para el split por episodio) y agrega el target `estado` heredado del
	episodio; ninguna de las dos se trata como feature de entrada del modelo.
	"""
	dataframe = dataframe.sort_values(["episodio_id", "segundo"])
	window_id = dataframe["segundo"] // WINDOW_SIZE

	rows = []
	for (episodio_id, ventana), group in dataframe.groupby([dataframe["episodio_id"], window_id]):
		estados = group[TARGET_COLUMN].unique()
		if len(estados) != 1:
			raise ValueError(
				f"Episodio {episodio_id} mezcla mas de un estado en una misma ventana."
			)
		features = _compute_window_features(group)
		features["episodio_id"] = episodio_id
		features["ventana_id"] = ventana
		features[TARGET_COLUMN] = estados[0]
		rows.append(features)

	columns = ["episodio_id", "ventana_id"] + FEATURE_NAMES + [TARGET_COLUMN]
	return pd.DataFrame(rows, columns=columns)
