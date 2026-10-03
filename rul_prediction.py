"""Turbofan Remaining-Useful-Life (RUL) prediction on NASA C-MAPSS FD001.

Predicts how many cycles an engine has left before failure from multivariate
sensor time series, and flags engines entering a failure-risk window.

Validation: GroupKFold by engine on train (no engine appears in both folds),
then a held-out evaluation on the official test engines (last observed cycle
vs the official RUL file).
"""
import json
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             precision_score, recall_score, f1_score)
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

RUL_CAP = 125          # piecewise-linear RUL target (standard for C-MAPSS)
WARN_AT = 30           # "failure-risk window" in cycles
WIN = 10               # rolling window for temporal features
cols = ["unit", "cycle"] + [f"set{i}" for i in (1, 2, 3)] + [f"s{i}" for i in range(1, 22)]

train = pd.read_csv("train_FD001.txt", sep=r"\s+", header=None, names=cols)
test = pd.read_csv("test_FD001.txt", sep=r"\s+", header=None, names=cols)
rul_true = pd.read_csv("RUL_FD001.txt", header=None, names=["rul"])["rul"].values

# --- target ---
train["RUL"] = (train.groupby("unit")["cycle"].transform("max") - train["cycle"]).clip(upper=RUL_CAP)

# --- drop constant / near-constant sensors ---
sensors = [c for c in cols if c.startswith("s") and not c.startswith("set")]
const = [c for c in sensors if train[c].std() < 1e-3]
sensors = [c for c in sensors if c not in const]
print("Dropped constant sensors:", const, "| kept:", len(sensors))


def add_features(df):
    df = df.sort_values(["unit", "cycle"]).copy()
    g = df.groupby("unit")[sensors]
    roll_mean = g.rolling(WIN, min_periods=1).mean().reset_index(level=0, drop=True)
    roll_std = g.rolling(WIN, min_periods=2).std().reset_index(level=0, drop=True).fillna(0)
    roll_mean.columns = [f"{c}_mean{WIN}" for c in sensors]
    roll_std.columns = [f"{c}_std{WIN}" for c in sensors]
    return pd.concat([df, roll_mean, roll_std], axis=1)


train_f = add_features(train)
test_f = add_features(test)
feat = sensors + [f"{c}_mean{WIN}" for c in sensors] + [f"{c}_std{WIN}" for c in sensors] + ["cycle"]
print("Total features:", len(feat))

X, y, groups = train_f[feat].values, train_f["RUL"].values, train_f["unit"].values
test_last = test_f.groupby("unit").tail(1)
X_test = test_last[feat].values
y_test = np.minimum(rul_true, RUL_CAP)


def nasa_score(y_true, y_pred):
    d = y_pred - y_true
    return float(np.sum(np.where(d < 0, np.exp(-d / 13) - 1, np.exp(d / 10) - 1)))


models = {
    "Baseline (mean)": DummyRegressor(strategy="mean"),
    "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=10)),
    "Random Forest": RandomForestRegressor(n_estimators=80, max_depth=14, min_samples_leaf=5, n_jobs=-1, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8,
                            colsample_bytree=0.8, random_state=42, n_jobs=-1),
}

rows = []
gkf = GroupKFold(n_splits=3)
for name, m in models.items():
    cv_rmse = []
    for tr, va in gkf.split(X, y, groups):
        m.fit(X[tr], y[tr])
        cv_rmse.append(np.sqrt(mean_squared_error(y[va], m.predict(X[va]))))
    m.fit(X, y)
    p = np.clip(m.predict(X_test), 0, RUL_CAP)
    rmse = float(np.sqrt(mean_squared_error(y_test, p)))
    mae = float(mean_absolute_error(y_test, p))
    r2 = float(r2_score(y_test, p))
    # failure-risk warning: actual RUL <= WARN_AT vs predicted RUL <= WARN_AT
    yt, yp = (y_test <= WARN_AT).astype(int), (p <= WARN_AT).astype(int)
    rows.append({
        "Model": name,
        "CV_RMSE_(3-fold by engine)": round(float(np.mean(cv_rmse)), 2),
        "Test_RMSE": round(rmse, 2), "Test_MAE": round(mae, 2), "Test_R2": round(r2, 3),
        "NASA_score": round(nasa_score(y_test, p), 0),
        "Warn_precision": round(float(precision_score(yt, yp, zero_division=0)), 3),
        "Warn_recall": round(float(recall_score(yt, yp, zero_division=0)), 3),
        "Warn_F1": round(float(f1_score(yt, yp, zero_division=0)), 3),
    })

res = pd.DataFrame(rows)
print(res.to_string(index=False))
res.to_csv("rul_model_comparison.csv", index=False)

best = res.loc[res["Test_RMSE"].idxmin()]
print("\nBest model by test RMSE:", best["Model"])
print("Test engines:", len(y_test), "| failure-risk positives:", int((y_test <= WARN_AT).sum()))
json.dump({"n_train_rows": int(len(train)), "n_train_engines": int(train.unit.nunique()),
           "n_test_engines": int(len(y_test)), "n_features": len(feat),
           "dropped_sensors": const, "best": best.to_dict()}, open("summary.json", "w"), indent=2, default=str)
