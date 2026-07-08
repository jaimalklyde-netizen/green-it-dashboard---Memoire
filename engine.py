from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import pandas as pd
import joblib
import os

app = FastAPI(title="GreenOps API", version="2.0")

class ServerData(BaseModel):
    CPU_Usage: float
    Memory_Capacity: float
    CPU_Cores: int
    Hardware_Year: int
    TDP_Limit: float

# Charger le modèle
model_path = 'moteur_prediction_xgboost.pkl'
if os.path.exists(model_path):
    model = joblib.load(model_path)
    print("✅ Modèle chargé")
else:
    raise FileNotFoundError("❌ Modèle introuvable")

def add_interaction_features(df):
    df = df.copy()
    df["CPU_RAM_interaction"] = (df["CPU_Usage (%)"] * df["Memory_Capacity (GB)"]) / 100
    df["Cores_Year_interaction"] = df["CPU_Cores"] * (2024 - df["Hardware_Year"])
    df["TDP_per_Core"] = df["TDP_Limit"] / df["CPU_Cores"]
    df["Efficiency_Score"] = (df["Hardware_Year"] / 2024) * (df["CPU_Cores"] / (df["TDP_Limit"] + 1))
    df["Density_Score"] = df["Memory_Capacity (GB)"] / df["CPU_Cores"]
    return df

@app.get("/")
def root():
    return {
        "message": "Bienvenue sur GreenOps API",
        "version": "2.0",
        "endpoints": {
            "POST /predict": "Prédiction pour un serveur unique",
            "GET /docs": "Documentation interactive (Swagger UI)",
            "GET /health": "Vérifier l'état de l'API"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict")
def predict(server: ServerData):
    try:
        df = pd.DataFrame([{
            "CPU_Usage (%)": server.CPU_Usage,
            "Memory_Capacity (GB)": server.Memory_Capacity,
            "CPU_Cores": server.CPU_Cores,
            "Hardware_Year": server.Hardware_Year,
            "TDP_Limit": server.TDP_Limit
        }])
        
        df = add_interaction_features(df)
        
        features = [
            "CPU_Usage (%)", "Memory_Capacity (GB)", "CPU_Cores",
            "Hardware_Year", "TDP_Limit",
            "CPU_RAM_interaction", "Cores_Year_interaction",
            "TDP_per_Core", "Efficiency_Score", "Density_Score"
        ]
        
        watts = float(model.predict(df[features])[0])
        
        kwh_per_year = (watts / 1000) * 8760
        co2_kg_per_year = kwh_per_year * 0.05
        cost_euros_per_year = kwh_per_year * 0.25
        
        return {
            "predicted_watts": round(watts, 2),
            "predicted_kwh_per_year": round(kwh_per_year, 2),
            "predicted_co2_kg_per_year": round(co2_kg_per_year, 2),
            "estimated_cost_euros_per_year": round(cost_euros_per_year, 2)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
