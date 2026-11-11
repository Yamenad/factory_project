# 🏭 Factory Sensor Simulator 2040 - Predictive Maintenance Project
## Guide de Compréhension Complète du Projet (Industry 5.0)

---

### 1. 🎯 Vue d'Ensemble du Projet

Ce projet a pour objectif de prédire la **Durée de Vie Restante** (*Remaining Useful Life* - **RUL**, exprimée en jours) d'équipements industriels à partir de données de capteurs en temps réel. 

Il s'inscrit dans un cadre **Industry 5.0**, combinant l'automatisation avancée et la supervision humaine assistée par IA.

* **Volume de données** : 500 000 enregistrements bruts, 22 colonnes (capteurs physiques, historiques de maintenance, métriques d'IA).
* **Technologie ML** : Réseau de neurones Perceptron Multi-Couches (**MLP**) développé en **PyTorch**.
* **Performance obtenue** :
  * **$R^2$** : $\approx 0.97$ (97% de la variance expliquée).
  * **MAE** : $\approx 38$ jours d'erreur moyenne absolue.
* **Interface Utilisateur** : Application de bureau interactive développée avec **PyQt6**.

---

### 2. 📁 Structure du Projet

```text
ml1/
│
├── data/
│   └── factory_sensor_simulator_2040.csv   # Dataset brut (500k lignes)
│
├── src/
│   ├── preprocess.py                        # Nettoyage et transformation des données
│   ├── train.py                             # Modèle PyTorch & boucle d'entraînement
│   ├── model.pth                            # Poids du réseau de neurones entraîné
│   └── preprocessor.joblib                  # Pipeline de prétraitement sauvegardé
│
├── app/
│   └── main_ui.py                           # Interface utilisateur PyQt6
│
├── presentation/
│   └── index.html                           # Simulation HTML/JS interactive
│
├── train.ipynb                              # Notebook explicatif pas à pas
├── requirements.txt                         # Dépendances Python du projet
└── README.md                                # Documentation générale
```

---

### 3. ⚙️ Pipeline de Traitement des Données (`preprocess.py`)

#### A. Dimensions des Données
1. **Entrée Brute** : `(500000, 22)` — 500 000 lignes et 22 colonnes.
2. **Sortie Prétraitée** : `(400000, 51)` en entraînement / `(100000, 51)` en test.
   * **Division** : 80% Entraînement / 20% Test.
   * **Transformation (22 ➔ 51 colonnes)** : Passage de la variable catégorielle `Machine_Type` en encodage **One-Hot** (création d'une colonne binaire par type de machine).

#### B. Traitement par Étape
* **Suppression des doublons** sur `Machine_ID`.
* **Imputation des valeurs manquantes** :
  * Variables numériques : Remplacement par la *médiane*.
  * Variables catégorielles : Remplacement par la *valeur la plus fréquente*.
* **Élimination des valeurs aberrantes (Clipping)** :
  * Température : $[0, 150]^\circ\text{C}$
  * Vibrations : $[0, 40]\text{ mm/s}$
  * Bruit : $[20, 130]\text{ dB}$
  * Niveaux d'huile et liquide de refroidissement : $[0, 100]\%$
* **Normalisation (`StandardScaler`)** : Centrage et réduction des variables numériques ($\mu=0, \sigma=1$) pour garantir la convergence du réseau de neurones.

---

### 4. 🧠 Architecture du Modèle PyTorch (`src/train.py`)

Le modèle est un réseau de neurones profond de type **MLP Regressor** conçu pour prédire une valeur continue (la durée de vie restante en jours).

```text
Entrée (51 features)
       │
       ▼
 ┌───────────┐  Linear(51 ➔ 128) + BatchNorm1d + ReLU + Dropout(0.1)
 ├───────────┤  Linear(128 ➔ 64)  + BatchNorm1d + ReLU + Dropout(0.1)
 ├───────────┤  Linear(64 ➔ 32)   + BatchNorm1d + ReLU + Dropout(0.1)
 └───────────┘
       │
       ▼
 Sortie (1 valeur continue: RUL prédite)
```

#### Éléments Clés du Modèle :
* **BatchNorm1d** : Stabilise l'apprentissage et accélère la convergence.
* **ReLU** : Fonction d'activation introduisant la non-linéarité.
* **Dropout (0.1)** : Désactive aléatoirement 10% des neurones pour éviter le surapprentissage (*overfitting*).
* **Optimiseur** : `AdamW` (Taux d'apprentissage initial = $0.002$).
* **Scheduler** : `ReduceLROnPlateau` (réduit le taux d'apprentissage quand la perte stagne).
* **Early Stopping** : Arrête l'entraînement si la métrique de validation ne s'améliore plus.

---

### 5. 💻 Application Desktop (`app/main_ui.py`)

L'application **PyQt6** permet aux opérateurs de charger une configuration de machine ou de saisir manuellement les mesures des capteurs afin d'obtenir immédiatement :
1. La prédiction de la durée de vie restante (**RUL** en jours).
2. Un indicateur visuel de risque (Vert = Bon état, Orange = Maintenance recommandée, Rouge = Danger imminent).
3. La possibilité de simuler des pannes en faisant varier des valeurs clés comme la température ou le niveau de vibration.

---

### 6. 🚀 Guide d'Installation et d'Exécution

#### Étape 1 : Créer et activer l'environnement virtuel
```bash
# Création
python -m venv venv

# Activation (Windows CMD)
venv\Scripts\activate.bat

# Activation (Windows PowerShell)
venv\Scripts\Activate.ps1

# Activation (Mac / Linux)
source venv/bin/activate
```

#### Étape 2 : Installer les dépendances (`requirements.txt`)
```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Étape 3 : Lancer l'entraînement ou le Notebook
```bash
# Lancer l'entraînement direct du modèle (génère src/model.pth)
python -m src.train

# Ou ouvrir le notebook explicatif dans VS Code / Jupyter Lab
jupyter lab train.ipynb
```

#### Étape 4 : Lancer l'interface utilisateur PyQt6
```bash
python -m app.main_ui
```

---

### 7. 💡 Synthèse des Métriques et Interprétation

| Métrique | Valeur Obtenue | Signification |
| :--- | :--- | :--- |
| **$R^2$ Score** | **~0.97** | Le modèle explique 97% de la variabilité de la durée de vie. |
| **MAE** | **~38 jours** | En moyenne, la prédiction diffère de la réalité de 38 jours. |
| **RMSE** | **~48 jours** | Faible pénalité pour les erreurs extrêmes, modèle très stable. |

---

### 8. 📊 Lexique des Métriques et Paramètres d'Entraînement

* **Epoch (Époque) :** 
  C'est un cycle complet d'apprentissage. Une époque signifie que le réseau de neurones a vu et analysé l'intégralité de toutes les données d'entraînement une fois. Si vous définissez `epochs = 15`, le modèle va s'entraîner en passant 15 fois de suite sur les données pour affiner ses poids et s'améliorer progressivement.
* **Train MSE (Mean Squared Error - Erreur Quadratique Moyenne) :** 
  C'est la mesure de l'erreur du modèle sur les données d'entraînement. Le mot *"Squared"* (au carré) signifie que les grosses erreurs sont fortement pénalisées (une erreur de 10 jours devient 100 dans le calcul). Plus ce chiffre est bas, mieux le modèle a mémorisé/appris les tendances.
* **Val MAE (Mean Absolute Error - Erreur Absolue Moyenne de validation) :** 
  C'est l'erreur moyenne commise par le modèle sur les données de validation (les données de contrôle mises de côté pendant l'entraînement). Exprimée directement en jours, si vous lisez `Val MAE: 38.2`, cela signifie que sur les machines de validation, les prédictions du modèle se trompent en moyenne de 38 jours par rapport à leur vraie durée de vie restante.
* **$R^2$ (Coefficient de détermination) :** 
  C'est le score global de performance du modèle (généralement entre 0 et 1). Un $R^2$ de 0,97 (soit 97 %) est exceptionnel : cela signifie que votre modèle est capable d'expliquer 97 % des variations de la durée de vie des machines. C'est la preuve mathématique que le modèle est très performant.

---

### 9. 📋 Description Détaillée des Variables du Dataset (`factory_sensor_simulator_2040.csv`)

Le fichier de simulation contient **22 colonnes d'origine** qui décrivent l'état, l'historique et l'environnement des machines industrielles. On les classe en plusieurs catégories :

#### A. La Cible (Ce que l'on veut prédire)
* **`Remaining_Useful_Life_days` (RUL) :** C'est la cible (*target*). C'est le nombre exact de jours qu'il reste à la machine avant la panne. C'est cette valeur que le réseau de neurones (MLP) essaie de deviner.

#### B. Les caractéristiques de la machine et d'exploitation
* **`Machine_Type` :** Le type de machine (ex: CNC_Mill, Robot_Arm, Furnace...). C'est la seule variable textuelle (catégorielle) qui est transformée en 51 colonnes binaires via le One-Hot Encoding.
* **`Installation_Year` :** L'année d'installation de la machine (permet de déduire son ancienneté).
* **`Operational_Hours` :** Le nombre total d'heures de fonctionnement cumulées.

#### C. Les capteurs physiques en temps réel
* **`Temperature_C` :** La température de fonctionnement en degrés Celsius (une surchauffe accélère l'usure).
* **`Vibration_mms` :** L'intensité des vibrations en mm/s (un indicateur clé de défaillance mécanique).
* **`Sound_dB` :** Le niveau sonore émis par la machine en décibels.
* **`Oil_Level_pct` & `Coolant_Level_pct` :** Les pourcentages de niveaux d'huile et de liquide de refroidissement.
* **`Power_Consumption_kW` :** La consommation électrique en kilowatts.
* **`Hydraulic_Pressure_bar` :** La pression hydraulique.
* **`Coolant_Flow_L_min` :** Le débit du liquide de refroidissement (L/min).
* **`Laser_Intensity` & `Heat_Index` :** Indicateurs thermiques ou optiques spécifiques à certains types de machines.

#### D. L'historique de maintenance et les alertes
* **`Last_Maintenance_Days_Ago` :** Le nombre de jours écoulés depuis la dernière opération de maintenance.
* **`Maintenance_History_Count` :** Le nombre total de maintenances subies par le passé.
* **`Failure_History_Count` :** Le nombre de pannes majeures déjà enregistrées par le passé.
* **`Error_Codes_Last_30_Days` :** Le nombre de codes d'erreur renvoyés par l'automate au cours des 30 derniers jours.

#### E. L'écosystème IA de l'usine du futur (Industry 5.0)
* **`AI_Supervision` :** Indique si la machine est sous supervision directe d'une IA (valeur booléenne Vrai/Faux).
* **`AI_Override_Events` :** Le nombre de fois où l'IA a dû corriger ou reprendre le contrôle d'une sous-routine de la machine.