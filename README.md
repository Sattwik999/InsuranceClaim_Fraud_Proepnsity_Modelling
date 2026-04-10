#Insurance Claim Fraud Propensity Modelling

**Machine Learning–Based Fraud Propensity Analysis for Insurance Claims**

> Digital Assignment – 3 · Predictive Analytics  
> **Aditya Singh** (Reg. No. 23BCE1293) &nbsp;|&nbsp; **Sattwik Sarkar** (Reg. No. 23BCE1297)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Sattwik999/InsuranceClaim_Fraud_Proepnsity_Modelling/blob/main/InsuranceClaim_FraudDetection.ipynb)

---

## 📌 Project Overview

Insurance fraud is a costly problem that drives up premiums and erodes trust. This project builds a **fraud propensity classifier** — a model that assigns a probability of fraud to each insurance claim — using a variety of supervised Machine Learning techniques along with strategies to handle severe class imbalance.

The notebook walks through the complete ML pipeline, from raw data exploration all the way to benchmarking multiple models and deriving business-level interpretations.

---

## 📂 Repository Structure

```
InsuranceClaim_Fraud_Proepnsity_Modelling/
│
├── InsuranceClaim_FraudDetection.ipynb   # Main Colab notebook
├── insurance_claims.csv                  # Dataset (1 000 records, 39 features)
└── README.md
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| File | `insurance_claims.csv` |
| Rows | 1 000 (insurance claims) |
| Features | 39 (policyholder info, incident details, claim amounts) |
| Target | `fraud_reported` — **Y / N** (mapped to **1 / 0**) |

### Key Feature Groups

| Group | Example Features |
|---|---|
| Policyholder | `months_as_customer`, `age`, `policy_state`, `insured_sex`, `insured_education_level` |
| Financial | `policy_annual_premium`, `capital-gains`, `capital-loss`, `total_claim_amount` |
| Claim / Incident | `incident_type`, `collision_type`, `incident_severity`, `authorities_contacted` |
| Vehicle | `auto_make`, `auto_model`, `auto_year` |
| Engineered | `tenure_days`, `vehicle_age`, `claim_ratio`, `net_capital`, `incident_time_of_day` |

> **Class Imbalance**: ~75 % of claims are non-fraudulent (Class 0) and ~25 % are fraudulent (Class 1), making standard accuracy a poor metric — **F1 / F2 score** and **ROC-AUC** are used instead.

---

## 🚀 Getting Started (Google Colab)

1. Click the **Open in Colab** badge above, *or* upload the notebook manually to [colab.research.google.com](https://colab.research.google.com).
2. Upload `insurance_claims.csv` to the Colab session storage (or mount Google Drive and place the file there).
3. Run all cells top-to-bottom — the first code cell installs all required packages automatically:
   ```python
   !pip install imbalanced-learn xgboost --quiet
   !pip install ipywidgets --quiet
   ```
4. The interactive **threshold-tuning slider** (Cell ~55) requires the `ipywidgets` extension, which is enabled by default in Colab.

---

## 🔬 Notebook Walkthrough

### 1 · Data Predictors & Target Identification
Separates `fraud_reported` as the target (`y`) and drops identifier columns (`policy_number`, `incident_date`, `insured_zip`) from the feature matrix (`X`).

### 2 · Class Distribution Visualization
Side-by-side bar chart + pie chart showing the raw class imbalance before any resampling.

### 3 & 4 · Data Preprocessing & Transformation
| Step | What happens |
|---|---|
| **Noise handling** | Replaces `'?'` with `NaN`, then forward-fills missing values |
| **Date features** | Extracts `tenure_days` (policy age) and `vehicle_age` |
| **Financial ratios** | `claim_ratio`, `net_capital`, `vehicle_claim_prop` |
| **Behavioural flags** | `is_out_of_state` (policy state ≠ incident state) |
| **Time binning** | `incident_time_of_day` (Night / Morning / Afternoon / Evening / Late Night) |
| **Column pruning** | Drops high-cardinality & redundant columns (`incident_location`, `_c39`, etc.) |

A **ColumnTransformer** applies:
- `StandardScaler` on numeric columns
- `OneHotEncoder` (handle_unknown='ignore') on categorical columns

### 5 · Data Splitting
- 80 / 20 train-test split, **stratified** on the target
- 5-fold cross-validation (F1) on a baseline Random Forest to assess the bias-variance tradeoff

### 6 · Model Training, Validation & Testing

Five model families are explored:

| Model | Notes |
|---|---|
| Logistic Regression | Baseline — no class balancing |
| Decision Tree | Default hyperparameters |
| Random Forest | 100 estimators |
| XGBoost | `eval_metric='logloss'` |
| Balanced Random Forest | From `imbalanced-learn` |

### 7 · Imbalance Remedies

| Strategy | Implementation |
|---|---|
| **Hybrid sampling** | SMOTE + RandomUnderSampler (imbalanced-learn pipeline) |
| **SMOTETomek** | Combined over- and under-sampling |
| **Class weights** | `class_weight={0:1, 1:5}` in Random Forest |
| **Balanced RF** | `BalancedRandomForestClassifier` |
| **Threshold tuning** | Interactive `ipywidgets` slider + automated F2-score optimisation |

### 8 · Proposed Best Model — XGBoost + SMOTETomek + Threshold Tuning

```
XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.05,
    subsample=0.8,    colsample_bytree=0.8,
    scale_pos_weight=1, eval_metric='logloss'
)
```
Wrapped in an `ImbPipeline` with `SMOTETomek`, followed by automated F2-score–optimal threshold search.

### 9 · Benchmarking All Models
- Combined ROC curve for all trained models
- Metrics table: Accuracy, Precision, Recall, F1, ROC-AUC
- F1 improvement from baseline to the proposed model (absolute + %)

### 10 · Business Interpretation
- **Feature importance** bar chart (top-10 fraud predictors from the Random Forest)
- **Propensity score box plot** — distribution of predicted fraud probability split by actual label
- **Clustering validation** — KMeans (k=2) silhouette score on encoded features

---

## 📈 Key Results (Illustrative)

| Model | F1 (Fraud) | ROC-AUC |
|---|---|---|
| Baseline Logistic Regression | ~0.45 | ~0.72 |
| Hybrid RF (SMOTE + Undersample) | ~0.60 | ~0.82 |
| **XGBoost + SMOTETomek + Threshold** | **~0.70+** | **~0.88+** |

> Exact numbers will vary with each Colab run due to sampling randomness. Set `random_state=42` (already done in the notebook) for reproducibility.

---

## 🛠️ Dependencies

| Package | Purpose |
|---|---|
| `pandas`, `numpy` | Data manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | Preprocessing, models, metrics |
| `imbalanced-learn` | SMOTE, SMOTETomek, Balanced RF |
| `xgboost` | Gradient-boosted tree classifier |
| `ipywidgets` | Interactive threshold slider |

All packages are installed inside the notebook via `!pip install` — no manual setup required in Colab.

---

## 📝 Metrics Glossary

| Metric | Why it matters here |
|---|---|
| **Precision** | Fraction of flagged claims that are truly fraudulent (cost of false alarms) |
| **Recall** | Fraction of actual frauds that are caught (cost of missed fraud) |
| **F1 Score** | Harmonic mean of Precision & Recall |
| **F2 Score** | Weights Recall twice as heavily — used for threshold tuning since missing fraud is more costly than a false alarm |
| **ROC-AUC** | Model's overall discriminative power across all thresholds |

---

## 👥 Authors

| Name | Registration No. |
|---|---|
| Aditya Singh | 23BCE1293 |
| Sattwik Sarkar | 23BCE1297 |
