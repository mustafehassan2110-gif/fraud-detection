# Mobile Money Fraud Detection: A Dollar-Value Business Case

> **Headline result (fill in after running):** On a held-out future period, the best model
> catches **X%** of fraud value and would have prevented an estimated **$X** in losses
> versus no model, after accounting for the cost of false alarms.

![Net value vs threshold](reports/figures/net_value_hist_gradient_boosting.png)

## 1. The Problem
*(2-4 sentences. Who loses money? What does a missed fraud cost? What does a false alarm cost?)*

- **Business goal:** catch as much fraud value as possible without drowning customers in false alarms.
- **Success metric:** net value = fraud amount caught − (cost per false alarm × number of false alarms).
- **Assumption:** each false alarm costs **$5** (support call + customer friction). *(Justify or change this.)*

## 2. The Data
- **Source:** PaySim synthetic mobile money dataset (Kaggle) - link here.
- **Size:** ~6.3M transactions over 743 hourly steps; fraud rate ≈ 0.13%.
- **Limitations:** synthetic data; results will not transfer 1:1 to a real provider.

**Key findings from EDA** *(add 3-4 charts from `notebooks/01_eda.ipynb`)*
- Fraud occurs only in `TRANSFER` and `CASH_OUT`.
- *(your finding)*

**Preparation decisions**
| Decision | Why |
|---|---|
| Dropped `isFlaggedFraud` | Output of the provider's own rule, would leak the label |
| Dropped account name columns | Near-unique IDs, no generalisable signal |
| Kept only TRANSFER / CASH_OUT | The only types where fraud occurs |
| Time-based split (60/20/20) | Mimics deployment; a random split would leak future patterns |

## 3. Approach
1. **Features:** balance-error features, "account emptied" flag, hour of day, log amount.
2. **Models:** logistic regression (baseline) → random forest → gradient boosting.
3. **Imbalance:** class weights. *(Add: did SMOTE help? Show the comparison.)*
4. **Threshold:** chosen on the validation set to maximise net value, then frozen for the test set.

## 4. Results
| Model | PR-AUC | Recall @ 90% precision | Net value ($) |
|---|---|---|---|
| Logistic regression (baseline) | | | |
| Random forest | | | |
| Gradient boosting | | | |

![PR curves](reports/figures/pr_curves.png)

**What drives the predictions?** *(SHAP summary + 2-3 sentences in plain language.)*

**Error analysis:** *(Which frauds were missed and why? Which legitimate transactions were flagged?)*

## 5. Limitations & Next Steps
- Synthetic data, single time window, no customer history features.
- Next: per-account velocity features, cost-sensitive learning, drift monitoring.

## 6. Reproduce
```bash
git clone https://github.com/<your-username>/fraud-detection.git
cd fraud-detection
pip install -r requirements.txt
# Download PaySim from Kaggle into data/
python -m src.train --data data/<paysim-file>.csv --cost-false-alarm 5 --shap
```

Quick check without the real data:
```bash
python -m src.make_sample_data --out data/sample.csv
python -m src.train --data data/sample.csv
```

## Project structure
```
├── README.md
├── requirements.txt
├── notebooks/        # 01_eda, 02_modeling, 03_business_value
├── src/              # features.py, evaluate.py, train.py
└── reports/          # results.json and figures
```
