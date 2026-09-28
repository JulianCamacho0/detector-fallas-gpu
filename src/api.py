from pathlib import Path

import joblib
from src.config import MODEL_FILE
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.schema import validate_data
from src.features import compute_window_features

app = FastAPI(
    title="API de Detección de Fallas en GPU",
    description="API para predecir el estado de la GPU a partir de telemetría agregada por ventana.",
    version="1.0.0"
)

model = joblib.load(MODEL_FILE)

class Lectura(BaseModel):
    temp_c: float
    power_w: float
    util_pct: float
    clock_mhz: float
    ecc_errors: int

class TelemetryRequest(BaseModel):
    lecturas: list[Lectura]

@app.post("/predict")
def predecir(request: TelemetryRequest):

    #Validar cantidad minima de registros
    if len(request.lecturas) < 10:
        raise HTTPException(status_code=400, detail="La ventana debe contener al menos 10 lecturas de telemetría.")

    df = pd.DataFrame([lectura.model_dump() for lectura in request.lecturas])

    try:
        features = pd.DataFrame([compute_window_features(df)])
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error al procesar los datos de telemetría: {str(e)}"
        )

    prediction = model.predict(features)[0]

    probabilities = model.predict_proba(features)[0]
    classes = model.classes_

    probability_by_class = {
        classes_name: float(probabilities)
        for classes_name, probabilities in zip(classes, probabilities)
    }

    confidence = probability_by_class.get(prediction, 0.0)

    return {
        "estado_predicho": prediction,
        "confianza": confidence
    }