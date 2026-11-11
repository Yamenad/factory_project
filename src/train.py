"""Train an MLP regressor (PyTorch) to predict Remaining_Useful_Life_days.

Usage (from the project root):
    python -m src.train                    # full dataset
    python -m src.train --sample 100000    # quick run
"""
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn

from src.preprocess import DATA_PATH, prepare

MODEL_PATH = Path(__file__).resolve().parent / "model.pth"


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden=(128, 64, 32), dropout: float = 0.1):
        super().__init__()
        layers, prev = [], in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


@torch.no_grad()
def predict(model, X, y_mean, y_std, device, batch=8192):
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device)
        out.append(model(xb).cpu().numpy())
    return np.clip(np.concatenate(out) * y_std + y_mean, 0, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA_PATH)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--sample", type=int, default=None, help="use N random rows only")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Preprocessing...")
    X_tr, X_te, y_tr, y_te = prepare(a.data, seed=a.seed, sample=a.sample)
    y_mean, y_std = float(y_tr.mean()), float(y_tr.std())
    print(f"features: {X_tr.shape[1]} | train: {len(X_tr)} | test: {len(X_te)} | device: {device}")

    # Hold out 10% of train for early stopping (test set stays untouched).
    n_val = int(0.1 * len(X_tr))
    X_val, y_val = X_tr[:n_val], y_tr[:n_val]
    X_fit, y_fit = X_tr[n_val:], y_tr[n_val:]

    Xt = torch.from_numpy(X_fit).to(device)
    yt = torch.from_numpy((y_fit - y_mean) / y_std).to(device)

    model = MLPRegressor(X_tr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = nn.MSELoss()

    best_mae, best_state, bad = float("inf"), None, 0
    for epoch in range(1, a.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(Xt), device=device)
        total = 0.0
        for i in range(0, len(Xt), a.batch):
            idx = perm[i:i + a.batch]
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            opt.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        val_mae = mean_absolute_error(y_val, predict(model, X_val, y_mean, y_std, device))
        sched.step(val_mae)
        print(f"epoch {epoch:02d} | train MSE {total / len(Xt):.4f} | val MAE {val_mae:.2f} d "
              f"| {time.time() - t0:.1f}s")
        if val_mae < best_mae - 1e-3:
            best_mae, best_state, bad = val_mae, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= a.patience:
                print("Early stopping.")
                break

    model.load_state_dict(best_state)
    pred = predict(model, X_te, y_mean, y_std, device)
    print("\n=== Test metrics ===")
    print(f"MAE : {mean_absolute_error(y_te, pred):.2f} days")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_te, pred)):.2f} days")
    print(f"R2  : {r2_score(y_te, pred):.4f}")

    torch.save({
        "state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "in_dim": int(X_tr.shape[1]),
        "hidden": [128, 64, 32],
        "y_mean": y_mean,
        "y_std": y_std,
    }, MODEL_PATH)
    print(f"Saved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
