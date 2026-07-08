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
