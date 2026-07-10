# ============================================================================
# GREENOPS DASHBOARD - API FASTAPI (engine.py)
# Version : 2.0 - Modèle Random Forest (STABLE)
# ============================================================================
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import os
import joblib
import logging
import sys

# === Configuration des logs ===
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

app = FastAPI(title="GreenOps API", version="2.0")

# === CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Fichiers statiques ===
app.mount("/static", StaticFiles(directory="web"), name="static")

# === Modèles Pydantic ===
class ServerData(BaseModel):
    CPU_Usage: float
    Memory_Capacity: float
    CPU_Cores: int
    Hardware_Year: int
    TDP_Limit: float
    server_name: Optional[str] = None

class AuditRequest(BaseModel):
    servers: List[ServerData]
    audit_name: str = "Audit"

class AuditPrediction(BaseModel):
    server_name: str = None
    predicted_watts: float
    annual_cost_euros: float
    annual_carbon_kgco2e: float

class AuditResponse(BaseModel):
    audit_name: str
    aggregate_kpis: dict
    predictions: List[AuditPrediction]

# === CHARGEMENT DU MODÈLE RANDOM FOREST ===
model_path = 'moteur_prediction_rf.pkl'

if os.path.exists(model_path):
    try:
        model = joblib.load(model_path)
        logger.info(f"✅ Modèle Random Forest chargé depuis {model_path}")
        logger.info(f"🔍 Type du modèle : {type(model)}")
        
        # Test de prédiction au démarrage
        test_df = pd.DataFrame([{
            "CPU_Usage (%)": 75,
            "Memory_Capacity (GB)": 64,
            "CPU_Cores": 16,
            "Hardware_Year": 2020,
            "TDP_Limit": 150
        }])
        
        test_features = [
            "CPU_Usage (%)",
            "Memory_Capacity (GB)",
            "CPU_Cores",
            "Hardware_Year",
            "TDP_Limit"
        ]
        
        test_pred = float(model.predict(test_df[test_features])[0])
        logger.info(f"🧪 PRÉDICTION DE TEST : {test_pred:.2f} W")
        if test_pred > 0:
            logger.info("✅ Le modèle fonctionne correctement !")
        else:
            logger.warning(f"⚠️ ATTENTION : Le modèle prédit {test_pred}")
            
    except Exception as e:
        logger.error(f"❌ ERREUR : {str(e)}")
        raise
else:
    raise FileNotFoundError(f"❌ Modèle introuvable : {model_path}")

# === ENDPOINTS ===
@app.get("/")
def root():
    return {
        "message": "Bienvenue sur GreenOps API",
        "version": "2.0",
        "endpoints": {
            "POST /predict": "Prédiction pour un serveur unique",
            "POST /audit": "Audit complet d'un parc multi-serveurs",
            "GET /dashboard": "Interface utilisateur",
            "GET /docs": "Documentation Swagger",
            "GET /health": "État de l'API"
        }
    }

@app.get("/dashboard")
async def dashboard():
    return FileResponse("web/index.html")

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
        
        features = [
            "CPU_Usage (%)",
            "Memory_Capacity (GB)",
            "CPU_Cores",
            "Hardware_Year",
            "TDP_Limit"
        ]
        
        watts = float(model.predict(df[features])[0])
        if watts < 0:
            watts = 0.0
        
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
        logger.error(f"❌ Erreur predict : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/audit", response_model=AuditResponse)
def audit(request: AuditRequest):
    try:
        results = []
        total_watts = 0
        total_cost = 0
        total_carbon = 0
        
        for i, server in enumerate(request.servers):
            df = pd.DataFrame([{
                "CPU_Usage (%)": server.CPU_Usage,
                "Memory_Capacity (GB)": server.Memory_Capacity,
                "CPU_Cores": server.CPU_Cores,
                "Hardware_Year": server.Hardware_Year,
                "TDP_Limit": server.TDP_Limit
            }])
            
            features = [
                "CPU_Usage (%)",
                "Memory_Capacity (GB)",
                "CPU_Cores",
                "Hardware_Year",
                "TDP_Limit"
            ]
            
            watts = float(model.predict(df[features])[0])
            if watts < 0:
                watts = 0.0
            
            kwh_per_year = (watts / 1000) * 8760
            cost = kwh_per_year * 0.25
            carbon = kwh_per_year * 0.05
            
            total_watts += watts
            total_cost += cost
            total_carbon += carbon
            
            results.append(AuditPrediction(
                server_name=server.server_name or f"Serveur {i+1}",
                predicted_watts=round(watts, 2),
                annual_cost_euros=round(cost, 2),
                annual_carbon_kgco2e=round(carbon, 2)
            ))
        
        return AuditResponse(
            audit_name=request.audit_name,
            aggregate_kpis={
                "total_predicted_watts": round(total_watts, 2),
                "total_annual_cost_euros": round(total_cost, 2),
                "total_annual_carbon_kgco2e": round(total_carbon, 2),
                "number_of_servers": len(request.servers)
            },
            predictions=results
        )
        
    except Exception as e:
        logger.error(f"❌ Erreur audit : {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
