# GreenOps Dashboard - Prédiction de l'empreinte carbone des infrastructures Cloud

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.6-green)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.1.1-orange)](https://xgboost.readthedocs.io/)

## 📊 Présentation

**GreenOps Dashboard** est un prototype de tableau de bord prédictif permettant d'estimer la consommation énergétique (en Watts) d'un parc de serveurs à partir de métriques d'usage (CPU, RAM) et de variables matérielles (cœurs, année, TDP).

### 🎯 Objectif

Répondre aux exigences de la directive **CSRD (Corporate Sustainability Reporting Directive)** en fournissant une méthode **agnostique, transparente et auditable** pour quantifier l'empreinte carbone du Scope 3 liée aux infrastructures Cloud.

### 🧠 Modèle

- **Algorithme** : XGBoost (Gradient Boosting)
- **Features** : 10 variables (5 originales + 5 d'interaction)
- **Performance** :
  - MAE : **14,40 W**
  - R² : **0,9835**
  - R² CV (5-Folds) : **0,9797 ± 0,0014**
- **Jeu de données** : SPECpower_ssj2008 (6 699 observations)

## 🚀 Installation

```bash
git clone https://github.com/jaimalklyde-netizen/green-it-dashboard---Memoire.git
cd green-it-dashboard---Memoire
pip install -r requirements.txt
python engine.py

L'API sera disponible à : http://localhost:8000

Documentation interactive Swagger : http://localhost:8000/docs

📁 Structure
text
green-it-dashboard---Memoire/
├── engine.py                      # API FastAPI
├── requirements.txt               # Dépendances
├── moteur_prediction_xgboost.pkl  # Modèle XGBoost
├── echantillon_serveurs.csv       # Données de test
├── web/                           # Interface utilisateur
└── README.md
📈 Performance
Métrique	Valeur
MAE	14,40 W
R²	0,9835
R² CV (5-Folds)	0,9797 ± 0,0014
Feature Importance (Top 5)
Variable	Importance
Cores_Year_interaction	53%
CPU_RAM_interaction	12%
CPU_Cores	5%
CPU_Usage (%)	3%
TDP_per_Core	2%
📝 Auteur
Klyde Moussipi | Master 2 Data Science & BI - Epitech Digital School | 2025-2026
