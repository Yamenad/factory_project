"""Data cleaning, imputation, scaling and train/test split for the Factory Sensor dataset."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "factory_sensor_simulator_2040.csv"
PREPROCESSOR_PATH = ROOT / "src" / "preprocessor.joblib"

TARGET = "Remaining_Useful_Life_days"
CATEGORICAL = ["Machine_Type"]
NUMERIC = [
    "Installation_Year", "Operational_Hours", "Temperature_C", "Vibration_mms",
    "Sound_dB", "Oil_Level_pct", "Coolant_Level_pct", "Power_Consumption_kW",
    "Last_Maintenance_Days_Ago", "Maintenance_History_Count", "Failure_History_Count",
    "AI_Supervision", "Error_Codes_Last_30_Days", "AI_Override_Events",
    # Sparse sensors (>90% missing): median-imputed
    "Laser_Intensity", "Hydraulic_Pressure_bar", "Coolant_Flow_L_min", "Heat_Index",
]
# Dropped on purpose: Machine_ID (identifier) and Failure_Within_7_Days (derived from the target -> leakage).

# Physical limits used to clip simulator noise (e.g. negative power or vibration).
CLIP_RANGES = {
    "Temperature_C": (0, 150), "Vibration_mms": (0, 40), "Sound_dB": (20, 130),
    "Oil_Level_pct": (0, 100), "Coolant_Level_pct": (0, 100),
    "Power_Consumption_kW": (0, 600), "Coolant_Flow_L_min": (0, 100),
}


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy containing only model features (plus target if present)."""
    df = df.copy()
    if "Machine_ID" in df.columns:
        df = df.drop_duplicates(subset="Machine_ID")
    if TARGET in df.columns:
        df = df.dropna(subset=[TARGET])
        df[TARGET] = df[TARGET].clip(lower=0)
    for col in NUMERIC:
        if col not in df.columns:
            df[col] = np.nan
    df["AI_Supervision"] = df["AI_Supervision"].map(
        lambda v: np.nan if pd.isna(v) else float(str(v).lower() in ("true", "1", "1.0"))
    )
    for col, (lo, hi) in CLIP_RANGES.items():
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(lo, hi)
    cols = CATEGORICAL + NUMERIC + ([TARGET] if TARGET in df.columns else [])
    return df[cols]


def build_preprocessor() -> ColumnTransformer:
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num, NUMERIC), ("cat", cat, CATEGORICAL)], sparse_threshold=0.0
    )


def prepare(csv_path: Path = DATA_PATH, test_size: float = 0.2, seed: int = 42,
            sample: int | None = None):
    """Load, clean, split, fit the preprocessor on the train set only, and save it."""
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}\nSee data/PUT_CSV_HERE.txt")
    df = pd.read_csv(csv_path)
    if sample and sample < len(df):
        df = df.sample(sample, random_state=seed)
    df = clean(df)
    X, y = df.drop(columns=[TARGET]), df[TARGET].to_numpy(dtype=np.float32)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
    pre = build_preprocessor().fit(X_tr)
    joblib.dump(pre, PREPROCESSOR_PATH)
    return (pre.transform(X_tr).astype(np.float32), pre.transform(X_te).astype(np.float32),
            y_tr, y_te)


def load_preprocessor() -> ColumnTransformer:
    if not PREPROCESSOR_PATH.exists():
        raise FileNotFoundError("preprocessor.joblib missing - run: python -m src.train")
    return joblib.load(PREPROCESSOR_PATH)


def machine_types() -> list[str]:
    return list(load_preprocessor().named_transformers_["cat"]["onehot"].categories_[0])


def transform_input(values: dict, pre: ColumnTransformer | None = None) -> np.ndarray:
    """Transform one raw sensor reading (dict) into the model input vector (1, n_features)."""
    pre = pre or load_preprocessor()
    row = {k: values.get(k, np.nan) for k in CATEGORICAL + NUMERIC}
    df = clean(pd.DataFrame([row]))
    return pre.transform(df).astype(np.float32)


if __name__ == "__main__":
    Xtr, Xte, ytr, yte = prepare()
    print(f"train {Xtr.shape} | test {Xte.shape} | saved {PREPROCESSOR_PATH.name}")
