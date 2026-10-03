# Turbofan Engine Remaining-Useful-Life (RUL) Prediction

Predictive-maintenance model on NASA's C-MAPSS FD001 turbofan dataset. It estimates how many operating cycles an engine has left before failure from multivariate sensor time series, and flags engines entering a failure-risk window.

## Data
- NASA C-MAPSS FD001: 100 training engines run to failure (20,631 cycles) and 100 test engines with official RUL labels.
- 6 constant sensors dropped, 15 kept. RUL target capped at 125 cycles (standard piecewise-linear target).

## Approach
- 46 features: 15 sensors + 10-cycle rolling mean and std per sensor + cycle count.
- Models: mean baseline, Ridge, Random Forest, XGBoost.
- Validation: 3-fold GroupKFold by engine (no engine appears in train and validation), then held-out evaluation on the 100 official test engines (last observed cycle vs official RUL).
- Failure-risk warning: engine flagged when predicted RUL <= 30 cycles.

## Results (100 held-out test engines)
| Model | Test RMSE | MAE | R2 | NASA score | Warn precision | Warn recall | Warn F1 |
|---|---|---|---|---|---|---|---|
| Baseline (mean) | 41.94 | 34.83 | -0.095 | 33354 | 0.000 | 0.00 | 0.000 |
| Ridge | 21.26 | 16.48 | 0.719 | 1418 | 1.000 | 0.44 | 0.611 |
| Random Forest | 17.83 | 13.03 | 0.802 | 681 | 0.947 | 0.72 | 0.818 |
| XGBoost | 17.66 | 12.84 | 0.806 | 675 | 0.944 | 0.68 | 0.791 |

XGBoost lowers test RMSE by 58% versus the baseline. Random Forest gives the best failure-warning F1.

## Run
```
pip install pandas numpy scikit-learn xgboost
python rul_prediction.py
```
Place `train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt` in the same folder (NASA Prognostics Data Repository).

## Next steps
Sequence models (LSTM/1D-CNN), the other FD002-FD004 subsets with multiple operating conditions, and live-sensor streaming into the dashboard.

## Live demo (Streamlit)
`app.py` is an interactive dashboard: pick a test engine, see predicted vs actual RUL, the RUL curve over cycles, raw sensor trends, a failure-risk flag, and a fleet table of all 100 engines.
```
pip install -r requirements.txt
streamlit run app.py
```
Deploy free on Streamlit Community Cloud: push this folder (including the three NASA .txt files) to GitHub, then New app -> select the repo -> main file `app.py`.
