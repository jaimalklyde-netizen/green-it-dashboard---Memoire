from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import xgboost as xgb
import os
import json

app = FastAPI()

# ============================================================================
# CHARGEMENT DU MODÈLE - FORMAT NATIF XGBOOST (pas joblib)
# ============================================================================
# On charge via xgb.XGBRegressor().load_model(), qui utilise le format JSON
# natif et garantit la compatibilité entre versions de xgboost, contrairement
# à joblib.load() sur un objet pickled.

model_path = 'moteur_prediction_xgboost.json'
features_meta_path = 'model_features.json'

if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Modèle introuvable : {model_path}")

model = xgb.XGBRegressor()
model.load_model(model_path)
print(f"✅ Modèle chargé (xgboost version installée : {xgb.__version__})")

# Vérification de cohérence de version (avertissement, pas bloquant)
if os.path.exists(features_meta_path):
    with open(features_meta_path) as f:
        meta = json.load(f)
    trained_version = meta.get("xgboost_version")
    if trained_version and trained_version != xgb.__version__:
        print(
            f"⚠️  ATTENTION : modèle entraîné avec xgboost=={trained_version}, "
            f"mais version installée ici = {xgb.__version__}. "
            "Le format natif reste compatible, mais il est recommandé de "
            "figer la même version dans requirements.txt pour éviter toute "
            "dérive de comportement entre les versions."
        )
    FEATURES = meta["features"]
else:
    # Fallback si le fichier de métadonnées n'a pas été déployé
    FEATURES = [
        "CPU_Usage (%)", "Memory_Capacity (GB)", "CPU_Cores",
        "Hardware_Year", "TDP_Limit",
        "CPU_RAM_interaction", "Cores_Year_interaction",
        "TDP_per_Core", "Efficiency_Score", "Density_Score"
    ]


def add_interaction_features(df):
    df = df.copy()
    df["CPU_RAM_interaction"] = (df["CPU_Usage (%)"] * df["Memory_Capacity (GB)"]) / 100
    df["Cores_Year_interaction"] = df["CPU_Cores"] * (2024 - df["Hardware_Year"])
    df["TDP_per_Core"] = df["TDP_Limit"] / df["CPU_Cores"]
    df["Efficiency_Score"] = (df["Hardware_Year"] / 2024) * (df["CPU_Cores"] / (df["TDP_Limit"] + 1))
    df["Density_Score"] = df["Memory_Capacity (GB)"] / df["CPU_Cores"]
    return df


class ServerData(BaseModel):
    CPU_Usage: float
    Memory_Capacity: float
    CPU_Cores: int
    Hardware_Year: int
    TDP_Limit: float


@app.post("/predict")
def predict(server: ServerData):
    df = pd.DataFrame([{
        "CPU_Usage (%)": server.CPU_Usage,
        "Memory_Capacity (GB)": server.Memory_Capacity,
        "CPU_Cores": server.CPU_Cores,
        "Hardware_Year": server.Hardware_Year,
        "TDP_Limit": server.TDP_Limit
    }])

    df = add_interaction_features(df)

    watts = float(model.predict(df[FEATURES])[0])

    # Garde-fou physique : une consommation ne peut pas être négative.
    # XGBoost avec objectif 'reg:squarederror' n'est pas contraint à des
    # valeurs positives — pour un input très éloigné de la distribution
    # d'entraînement (ex: matériel plus récent que 2022, la borne max vue
    # à l'entraînement), une extrapolation légèrement négative reste possible
    # même avec un modèle correctement chargé. On clippe à 0 par sécurité.
    if watts < 0:
        print(f"⚠️  Prédiction négative brute ({watts:.2f} W) clippée à 0 pour input {server.dict()}")
        watts = 0.0

    return {"predicted_watts": watts}
