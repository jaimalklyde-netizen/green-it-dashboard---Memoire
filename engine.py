import pandas as pd

class GreenOpsEngine:
    """
    Moteur de calcul métier pour la traduction de l'empreinte énergétique
    en indicateurs financiers (FinOps) et écologiques (GreenOps).
    """
    def __init__(self, model):
        self.model = model
        
    def calculate_kpis(self, df, carbon_intensity=50, electricity_price=0.25):
        """
        Calcule les KPIs annuels sur la base des prédictions de l'IA.
        - carbon_intensity : gCO2e/kWh (Ex: 50 pour la France, très décarbonée)
        - electricity_price : Prix du kWh en euros
        """
        # 1. Prédiction des Watts en temps réel via le modèle IA
        X = df[['CPU_Usage (%)', 'Memory_Capacity (GB)', 'CPU_Cores', 'Hardware_Year', 'TDP_Limit']]
        df['Predicted_Watts'] = self.model.predict(X)
        
        # 2. Projection sur une année complète (8760 heures)
        df['Annual_Energy_kWh'] = (df['Predicted_Watts'] * 8760) / 1000
        
        # 3. Calculs Managériaux
        df['Annual_Cost_Euros'] = df['Annual_Energy_kWh'] * electricity_price
        df['Annual_Carbon_kgCO2e'] = (df['Annual_Energy_kWh'] * carbon_intensity) / 1000
        
        # 4. Détection des anomalies : Serveurs "Zombies" (Usage < 10%)
        zombies = df[df['CPU_Usage (%)'] < 10]
        
        # 5. Agrégation des résultats pour le Dashboard
        dashboard_metrics = {
            "total_servers": len(df),
            "total_energy_kwh": df['Annual_Energy_kWh'].sum(),
            "total_cost_euros": df['Annual_Cost_Euros'].sum(),
            "total_carbon_tons": df['Annual_Carbon_kgCO2e'].sum() / 1000, # Conversion en Tonnes
            "zombie_count": len(zombies),
            "wasted_cost_euros": zombies['Annual_Cost_Euros'].sum(),
            "wasted_carbon_tons": zombies['Annual_Carbon_kgCO2e'].sum() / 1000
        }
        
        return df, dashboard_metrics