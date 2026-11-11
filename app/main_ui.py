from __future__ import annotations
import sys
from pathlib import Path

# --- SOLUTION POUR L'EXÉCUTION DIRECTE ---
# Ajoute la racine du projet (ml1) au chemin Python pour éviter l'erreur "ModuleNotFoundError: No module named 'src'"
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
# ----------------------------------------

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QLineEdit, 
    QPushButton, QVBoxLayout, QFormLayout, QGroupBox, QMessageBox, QComboBox
)
from PyQt6.QtCore import Qt
import torch
import numpy as np

from src.preprocess import machine_types, transform_input, load_preprocessor
from src.train import MLPRegressor, MODEL_PATH

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Factory Sensor 2040 - RUL Prediction (Industry 5.0)")
        self.resize(650, 700)
        
        # Charger le modèle entraîné et le pré-traiteur
        try:
            self.device = torch.device("cpu")
            checkpoint = torch.load(MODEL_PATH, map_location=self.device)
            self.y_mean = checkpoint["y_mean"]
            self.y_std = checkpoint["y_std"]
            
            self.model = MLPRegressor(checkpoint["in_dim"], hidden=checkpoint["hidden"])
            self.model.load_state_dict(checkpoint["state_dict"])
            self.model.to(self.device)
            self.model.eval()
            self.preprocessor = load_preprocessor()
        except Exception as e:
            QMessageBox.critical(None, "Erreur", f"Impossible de charger le modèle. Avez-vous exécuté 'python -m src.train' ?\n\nErreur: {e}")
            sys.exit(1)

        self.init_ui()

    def init_ui(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        # Titre
        title_label = QLabel("Maintenance Prédictive - Capteurs d'Usine (Industry 5.0)")
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; color: #2C3E50;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title_label)

        # Formulaire de saisie des capteurs
        form_group = QGroupBox("Paramètres de la Machine et Capteurs")
        form_layout = QFormLayout()

        self.inputs = {}
        
        # Type de machine (Menu déroulant)
        self.machine_combo = QComboBox()
        try:
            self.machine_combo.addItems(machine_types())
        except:
            self.machine_combo.addItems(["CNC_Mill", "Furnace", "Robot_Arm", "Laser_Cutter", "Mixer", "Industrial_Chiller"])
        form_layout.addRow("Type de Machine :", self.machine_combo)

        # Champs numériques principaux
        fields = [
            ("Installation_Year", "2025"),
            ("Operational_Hours", "50000"),
            ("Temperature_C", "75.0"),
            ("Vibration_mms", "10.5"),
            ("Sound_dB", "80.0"),
            ("Oil_Level_pct", "50.0"),
            ("Coolant_Level_pct", "70.0"),
            ("Power_Consumption_kW", "120.0"),
            ("Last_Maintenance_Days_Ago", "30"),
            ("Maintenance_History_Count", "3"),
            ("Failure_History_Count", "1"),
            ("AI_Supervision", "1"),
            ("Error_Codes_Last_30_Days", "2"),
            ("AI_Override_Events", "1")
        ]

        for key, default_val in fields:
            txt_box = QLineEdit(default_val)
            form_layout.addRow(f"{key} :", txt_box)
            self.inputs[key] = txt_box

        form_group.setLayout(form_layout)
        main_layout.addWidget(form_group)

        # Bouton de prédiction
        predict_btn = QPushButton("Prédire la Durée de Vie Restante (RUL)")
        predict_btn.setStyleSheet("background-color: #27AE60; color: white; font-weight: bold; font-size: 14px; padding: 10px;")
        predict_btn.clicked.connect(self.make_prediction)
        main_layout.addWidget(predict_btn)

        # Affichage des KPIs de sortie
        kpi_group = QGroupBox("Résultats / KPIs")
        kpi_layout = QVBoxLayout()
        self.result_label = QLabel("Durée de vie estimée (RUL) : En attente de calcul...")
        self.result_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #2980B9;")
        kpi_layout.addWidget(self.result_label)
        kpi_group.setLayout(kpi_layout)
        main_layout.addWidget(kpi_group)

        self.setCentralWidget(central_widget)

    def make_prediction(self):
        try:
            # Récupérer les données du formulaire
            data = {"Machine_Type": self.machine_combo.currentText()}
            for key, widget in self.inputs.items():
                data[key] = float(widget.text())

            # Transformer l'entrée via le pré-traiteur
            x_transformed = transform_input(data, self.preprocessor)

            # Faire la prédiction avec le modèle PyTorch
            with torch.no_grad():
                xb = torch.from_numpy(x_transformed).to(self.device)
                pred_scaled = self.model(xb).item()
                rul_days = pred_scaled * self.y_std + self.y_mean
                rul_days = max(0.0, rul_days) # Pas de RUL négatif

            # Afficher le résultat
            status = "⚠️ Risque élevé de panne imminente !" if rul_days < 30 else "✅ Machine en bon état de fonctionnement."
            self.result_label.setText(f"Durée de vie restante estimée (RUL) : {rul_days:.1f} jours\nStatut : {status}")

        except Exception as e:
            QMessageBox.warning(self, "Erreur de saisie", f"Veuillez vérifier vos valeurs numériques.\n\nDétail: {e}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())