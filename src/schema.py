"""Contrato de validacion para la telemetria publica de GPUs NVIDIA L40.

Define el esquema esperado del DataFrame de entrada usando Pandera y
expone `validate_data` para validarlo antes de cualquier procesamiento.
"""

import pandas as pd
import pandera as pa

ESTADOS_VALIDOS = (
	"normal",
	"sobrecalentamiento",
	"degradacion_memoria",
	"falla_alimentacion",
)

TELEMETRIA_SCHEMA = pa.DataFrameSchema(
	columns={
		# Es un identificador de episodio, no una variable numerica para el modelo.
		"episodio_id": pa.Column(int, nullable=False),
		"segundo": pa.Column(int, checks=pa.Check.ge(0), nullable=False),
		# Limite amplio de plausibilidad del sensor; no es un umbral de clasificacion.
		# Incluye las lecturas de sobrecalentamiento del CSV (maximo observado: 99.2 C).
		"temp_c": pa.Column(
			float,
			checks=pa.Check.in_range(0, 100),
			nullable=False,
		),
		"power_w": pa.Column(float, checks=pa.Check.gt(0), nullable=False),
		"util_pct": pa.Column(
			float,
			checks=pa.Check.in_range(0, 100),
			nullable=False,
		),
		"clock_mhz": pa.Column(float, checks=pa.Check.ge(0), nullable=False),
		"ecc_errors": pa.Column(int, checks=pa.Check.ge(0), nullable=False),
		"estado": pa.Column(
			str,
			checks=pa.Check.isin(ESTADOS_VALIDOS),
			nullable=False,
		),
	},
	unique=["episodio_id", "segundo"],
	strict=True,
	coerce=False,
)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Valida `df` contra `TELEMETRIA_SCHEMA`.

    Devuelve el mismo DataFrame si es válido, o lanza
    `pandera.errors.SchemaError` con el detalle del incumplimiento.
    """
    return TELEMETRIA_SCHEMA.validate(df)
