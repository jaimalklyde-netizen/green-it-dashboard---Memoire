import pandas as pd
import joblib
import os

class GreenOpsEngine:
    """
    Moteur de calcul métier pour la traduction de l'empreinte énergétique
    en indicateurs financiers (FinOps) et écologiques (GreenOps).
    Version 2.0 : Modèle XGBoost + 10 features
    """
    def __init__(self, model_path='moteur_prediction_xgboost.pkl'):
        """Initialise le moteur avec le modèle XGBoost depuis le fichier .pkl"""
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            print(f"✅ Modèle XGBoost chargé depuis {model_path}")
        else:
            # Fallback sur l'ancien modèle Random Forest
            fallback_path = 'moteur_prediction_specpower.pkl'
            if os.path.exists(fallback_path):
                self.model = joblib.load(fallback_path)
                print(f"⚠️ Fallback sur Random Forest ({fallback_path})")
            else:
                raise FileNotFoundError(f"❌ Aucun modèle trouvé : {model_path} ou {fallback_path}")
    
    def _add_interaction_features(self, df):
        """Ajoute les 5 variables d'interaction"""
        df = df.copy()
        df["CPU_RAM_interaction"] = (df["CPU_Usage (%)"] * df["Memory_Capacity (GB)"]) / 100
        df["Cores_Year_interaction"] = df["CPU_Cores"] * (2024 - df["Hardware_Year"])
        df["TDP_per_Core"] = df["TDP_Limit"] / df["CPU_Cores"]
        df["Efficiency_Score"] = (df["Hardware_Year"] / 2024) * (df["CPU_Cores"] / (df["TDP_Limit"] + 1))
        df["Density_Score"] = df["Memory_Capacity (GB)"] / df["CPU_Cores"]
        return df
    
    def calculate_kpis(self, df, carbon_intensity=50, electricity_price=0.25):
        """Calcule les KPIs annuels sur la base des prédictions"""
        # 1. Feature Engineering (10 features)
        df_engineered = self._add_interaction_features(df)
        
        # 2. Sélection des 10 features
        features = [
            "CPU_Usage (%)",
            "Memory_Capacity (GB)",
            "CPU_Cores",
            "Hardware_Year",
            "TDP_Limit",
            "CPU_RAM_interaction",
            "Cores_Year_interaction",
            "TDP_per_Core",
            "Efficiency_Score",
            "Density_Score"
        ]
        X = df_engineered[features]
        
        # 3. Prédiction
        df['Predicted_Watts'] = self.model.predict(X)
        
        # 4. KPIs
        df['Annual_Energy_kWh'] = (df['Predicted_Watts'] * 8760) / 1000
        df['Annual_Cost_Euros'] = df['Annual_Energy_kWh'] * electricity_price
        df['Annual_Carbon_kgCO2e'] = (df['Annual_Energy_kWh'] * carbon_intensity) / 1000
        
        # 5. Serveurs zombies
        zombies = df[df['CPU_Usage (%)'] < 10]
        
        # 6. Dashboard metrics
        dashboard_metrics = {
            "total_servers": len(df),
            "total_energy_kwh": df['Annual_Energy_kWh'].sum(),
            "total_cost_euros": df['Annual_Cost_Euros'].sum(),
            "total_carbon_tons": df['Annual_Carbon_kgCO2e'].sum() / 1000,
            "zombie_count": len(zombies),
            "wasted_cost_euros": zombies['Annual_Cost_Euros'].sum(),
            "wasted_carbon_tons": zombies['Annual_Carbon_kgCO2e'].sum() / 1000,
            "model_used": "XGBoost"
        }
        
        return df, dashboard_metrics
