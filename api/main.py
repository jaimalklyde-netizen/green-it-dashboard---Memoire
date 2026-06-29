from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import joblib
from engine import GreenOpsEngine

# Initialisation de l'API
app = FastAPI(
    title="GreenOps API", 
    description="API de prédiction de consommation Cloud et pilotage ESG"
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

# Définition stricte du schéma de données attendu (Garantit que le Front-end n'envoie pas de données erronées)
class ServerData(BaseModel):
    cpu_usage: float
    memory_gb: float
    cpu_cores: int
    hardware_year: int
    tdp_limit: float

# Création du point d'entrée (Endpoint)
@app.post("/predict")
def predict_energy(data: ServerData):
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
            "predictions": {
                "predicted_watts": float(df_results['Predicted_Watts'].iloc[0]),
                "annual_cost_euros": float(df_results['Annual_Cost_Euros'].iloc[0]),
                "annual_carbon_kgco2e": float(df_results['Annual_Carbon_kgCO2e'].iloc[0])
            },
            "global_kpis": kpis
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de calcul interne : {str(e)}")