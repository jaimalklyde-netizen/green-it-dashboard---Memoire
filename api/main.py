from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
import pandas as pd
import joblib
from engine import GreenOpsEngine
from datetime import datetime

# Initialisation de l'API
app = FastAPI(
    title="GreenOps API", 
    description="API de prédiction de consommation Cloud et pilotage ESG - Support multi-serveurs"
)

# Configuration CORS (Indispensable pour que le Front-end HTML/JS puisse parler à l'API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chargement en mémoire du modèle IA et du moteur métier au démarrage du serveur
try:
    model = joblib.load('moteur_prediction_specpower.pkl')
    engine = GreenOpsEngine(model)
except Exception as e:
    print(f"Erreur critique lors du chargement des dépendances IA : {e}")

# ==================== SCHÉMAS DE DONNÉES ====================

# Schéma pour un serveur individuel
class ServerData(BaseModel):
    """Schéma de validation pour les données d'un serveur"""
    cpu_usage: float = Field(..., ge=0, le=100, description="Utilisation CPU en %")
    memory_gb: float = Field(..., gt=0, description="Capacité RAM en GB")
    cpu_cores: int = Field(..., gt=0, description="Nombre de cœurs CPU")
    hardware_year: int = Field(..., ge=2000, description="Année du matériel")
    tdp_limit: float = Field(..., gt=0, description="TDP limite en Watts")
    server_name: Optional[str] = Field(None, description="Nom optionnel du serveur")

# Schéma pour l'audit multi-serveurs
class AuditRequest(BaseModel):
    """Schéma pour une demande d'audit de parc"""
    servers: List[ServerData] = Field(..., min_items=1, description="Liste d'au moins 1 serveur")
    audit_name: Optional[str] = Field(None, description="Nom de l'audit")

# ==================== RÉPONSES ====================

class PredictionResponse(BaseModel):
    """Réponse pour une prédiction unique"""
    server_name: Optional[str]
    predicted_watts: float
    annual_cost_euros: float
    annual_carbon_kgco2e: float

class AuditResponse(BaseModel):
    """Réponse pour un audit multi-serveurs"""
    status: str
    timestamp: str
    audit_name: Optional[str]
    total_servers: int
    predictions: List[PredictionResponse]
    aggregate_kpis: dict

# ==================== ENDPOINTS ====================

@app.post("/predict", response_model=dict)
def predict_energy(data: ServerData):
    """
    Prédiction pour un serveur unique (rétrocompatibilité)
    """
    try:
        # 1. Formatage des données d'entrée pour Pandas
        input_data = pd.DataFrame([{
            'CPU_Usage (%)': data.cpu_usage,
            'Memory_Capacity (GB)': data.memory_gb,
            'CPU_Cores': data.cpu_cores,
            'Hardware_Year': data.hardware_year,
            'TDP_Limit': data.tdp_limit
        }])

        # 2. Exécution du moteur métier (Calcul des Watts, Euros et CO2)
        df_results, kpis = engine.calculate_kpis(input_data)
        
        # 3. Renvoi de la réponse structurée en JSON
        return {
            "status": "success",
            "server_name": data.server_name or "Serveur unique",
            "predictions": {
                "predicted_watts": float(df_results['Predicted_Watts'].iloc[0]),
                "annual_cost_euros": float(df_results['Annual_Cost_Euros'].iloc[0]),
                "annual_carbon_kgco2e": float(df_results['Annual_Carbon_kgCO2e'].iloc[0])
            },
            "global_kpis": kpis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de calcul interne : {str(e)}")


@app.post("/audit", response_model=dict)
async def audit_park(audit: AuditRequest):
    """
    Audit complet du parc serveurs (multi-serveurs)
    
    Accepte une liste de serveurs et retourne :
    - Les prédictions individuelles
    - Les KPIs agrégés du parc
    - Les statistiques globales
    """
    try:
        predictions_list = []
        total_watts = 0
        total_cost = 0
        total_carbon = 0
        
        # Boucle sur chaque serveur
        for server in audit.servers:
            # 1. Formatage des données
            input_data = pd.DataFrame([{
                'CPU_Usage (%)': server.cpu_usage,
                'Memory_Capacity (GB)': server.memory_gb,
                'CPU_Cores': server.cpu_cores,
                'Hardware_Year': server.hardware_year,
                'TDP_Limit': server.tdp_limit
            }])

            # 2. Exécution du moteur métier
            df_results, kpis = engine.calculate_kpis(input_data)
            
            # 3. Extraction des résultats
            predicted_watts = float(df_results['Predicted_Watts'].iloc[0])
            annual_cost = float(df_results['Annual_Cost_Euros'].iloc[0])
            annual_carbon = float(df_results['Annual_Carbon_kgCO2e'].iloc[0])
            
            # 4. Accumulation des totaux
            total_watts += predicted_watts
            total_cost += annual_cost
            total_carbon += annual_carbon
            
            # 5. Ajout à la liste des résultats
            predictions_list.append(PredictionResponse(
                server_name=server.server_name or f"Serveur_{len(predictions_list) + 1}",
                predicted_watts=predicted_watts,
                annual_cost_euros=annual_cost,
                annual_carbon_kgco2e=annual_carbon
            ))
        
        # Calcul des statistiques du parc
        num_servers = len(audit.servers)
        avg_watts = total_watts / num_servers if num_servers > 0 else 0
        avg_cost = total_cost / num_servers if num_servers > 0 else 0
        avg_carbon = total_carbon / num_servers if num_servers > 0 else 0
        
        # KPIs agrégés
        aggregate_kpis = {
            "total_predicted_watts": round(total_watts, 2),
            "average_watts_per_server": round(avg_watts, 2),
            "total_annual_cost_euros": round(total_cost, 2),
            "average_cost_per_server_euros": round(avg_cost, 2),
            "total_annual_carbon_kgco2e": round(total_carbon, 2),
            "average_carbon_per_server_kgco2e": round(avg_carbon, 2),
            "co2_reduction_potential_percent": 15,  # À adapter selon votre métier
            "cost_optimization_potential_percent": 20  # À adapter selon votre métier
        }
        
        # Réponse complète
        response = AuditResponse(
            status="success",
            timestamp=datetime.now().isoformat(),
            audit_name=audit.audit_name or f"Audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            total_servers=num_servers,
            predictions=predictions_list,
            aggregate_kpis=aggregate_kpis
        )
        
        return response.dict()
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'audit du parc : {str(e)}")


@app.get("/health")
def health_check():
    """
    Endpoint de santé pour vérifier que l'API est opérationnelle
    """
    return {
        "status": "healthy",
        "service": "GreenOps API",
        "timestamp": datetime.now().isoformat(),
        "model_loaded": model is not None,
        "engine_initialized": engine is not None
    }


@app.get("/")
def root():
    """
    Route racine avec documentation des endpoints
    """
    return {
        "message": "Bienvenue sur GreenOps API",
        "version": "2.0",
        "endpoints": {
            "POST /predict": "Prédiction pour un serveur unique",
            "POST /audit": "Audit complet d'un parc multi-serveurs",
            "GET /health": "Vérifier l'état de l'API",
            "GET /docs": "Documentation interactive (Swagger UI)",
            "GET /redoc": "Documentation alternative (ReDoc)"
        }
    }