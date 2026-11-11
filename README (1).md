# Industry 5.0 – Maintenance prédictive par réseau de neurones

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-MLP-EE4C2C?logo=pytorch&logoColor=white)
![PyQt](https://img.shields.io/badge/UI-PyQt-41CD52?logo=qt&logoColor=white)

Mini-projet de Machine Learning : prédire la **durée de vie restante (RUL, *Remaining Useful Life*)** d'un équipement industriel à partir de mesures de capteurs, afin de planifier la maintenance avant la panne.

Les données proviennent d'une **simulation de capteurs d'usine Industrie 5.0**. Un **perceptron multicouche (MLP) PyTorch** est entraîné pour estimer la RUL en jours, et une **application de bureau PyQt** permet de saisir des mesures et d'obtenir la prédiction. Une **page web interactive** illustre le fonctionnement du réseau.

## Objectifs

- Construire un pipeline complet : nettoyage → normalisation → modèle → prédiction.
- Entraîner et évaluer un modèle de régression de la RUL.
- Fournir une interface exploitable par un opérateur de maintenance.
- Expliquer visuellement le fonctionnement du MLP (`presentation/index.html`).

## Pipeline

| Étape | Rôle |
|---|---|
| 1. Nettoyage | Gestion des valeurs manquantes, des aberrations et des doublons |
| 2. Normalisation | Mise à l'échelle des capteurs (paramètres calculés sur le jeu d'entraînement) |
| 3. MLP (PyTorch) | Couches denses + activations non linéaires, entraînement par rétropropagation |
| 4. Prédiction | RUL estimée en jours, affichée dans l'application PyQt |

## Architecture du projet

> À adapter si l'arborescence réelle diffère.

```
.
├── app/
│   └── main_ui.py          # Application de bureau PyQt
├── data/                   # Dataset de simulation des capteurs
├── models/                 # Modèle entraîné et paramètres de normalisation
├── presentation/
│   └── index.html          # Page interactive : pipeline du MLP
├── src/
│   ├── __init__.py
│   ├── ...                 # Prétraitement, modèle, évaluation
│   └── train.py            # Entraînement du modèle
├── requirements.txt
└── README.md
```

## Installation

```bash
git clone https://github.com/Yamenad/<nom-du-depot>.git
cd <nom-du-depot>

# (recommandé) environnement virtuel
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate

pip install -r requirements.txt
```

## Utilisation

### 1. Entraîner le modèle

```bash
python -m src.train
```

### 2. Lancer l'application de bureau (PyQt)

```bash
python -m app.main_ui
# ou
python app/main_ui.py
```

### 3. Ouvrir la présentation interactive

Ouvrir `presentation/index.html` dans un navigateur (aucun serveur requis). La page simule l'injection de mesures de capteurs et montre leur propagation dans le réseau jusqu'à la RUL.

## Résultats

Performances obtenues sur le jeu de test :

| Métrique | Valeur |
|---|---|
| **R²** | ≈ 0,97 |
| **MAE** | ≈ 38 jours |

Le modèle explique environ 97 % de la variance de la RUL, avec une erreur absolue moyenne d'environ 38 jours.

## Technologies

- **Python**, **PyTorch**
- **PyQt**
- **HTML / CSS / JavaScript** 

## Auteur

**Aouadhi Yamen** 
