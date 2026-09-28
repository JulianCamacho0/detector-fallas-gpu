"""Entrenamiento del modelo de clasificación de tickets."""

from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(
    random_state=42,
    n_estimators=200,
    n_jobs=1
)

def train_model(X_train, y_train) -> RandomForestClassifier:
    """Entrena el modelo de clasificación de tickets."""
    rf_model.fit(X_train, y_train)
    return rf_model