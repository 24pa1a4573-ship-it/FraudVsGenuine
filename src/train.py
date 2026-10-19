"""
train.py
────────────────────────────────────────────────────────────────
Full training pipeline for the Returns Fraud vs. Genuine
Grievance Sorter system.

Steps:
  1. Load processed returns dataset
  2. Customer-grouped train / val / test split (GroupShuffleSplit)
  3. Feature engineering pipeline (tabular)
  4. LightGBM behavioural model training with cost-sensitive
     class weights
  5. TF-IDF + Calibrated Logistic Regression NLP model for
     complaint text risk
  6. SHAP explainer serialization
  7. Cost-optimized threshold search (5:1 FP:FN cost ratio)
  8. Full metric evaluation: PR-AUC, ROC-AUC, confusion matrix

Usage:
    python src/train.py
Outputs (saved to models/):
    lgbm_model.pkl
    nlp_pipeline.pkl
    shap_explainer.pkl
    feature_columns.json
    thresholds.json
    training_report.txt
"""

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
FP_COST      = 5      # False Positive costs 5× more than FN
FN_COST      = 1

DATA_PATH    = Path("data/processed/returns_dataset.csv")
MODELS_DIR   = Path("models")

# Tabular features used by LightGBM
TABULAR_FEATURES = [
    "days_to_return",
    "days_to_return_ratio",
    "freight_ratio",
    "is_late_delivery_real",
    "price_vs_category_median",
    "customer_historical_return_rate",
    "product_category_return_propensity",
    "payment_installments",
    "carrier_confirmed_delivery",
    "is_high_ticket",
    "is_fit_sensitive",
    "is_fragile_category",
    "account_age_days",
    "delivery_delay_days",
    "review_score",
    "num_items",
    "total_price",
    "freight_ratio",
]

# NLP features
NLP_TEXT_COL  = "complaint_text"
NLP_LABEL_COL = "nlp_label"

# NLP categories mapping (for human-readable output)
NLP_LABEL_NAMES = {
    0: "Defective",
    1: "Wrong Item",
    2: "Transit Damage",
    3: "False INR",
    4: "Vague/Remorse",
}

# NLP risk mapping: how "fraudy" is each complaint category?
NLP_RISK_MAP = {
    0: 0.05,   # Defective      → low risk (genuine signal)
    1: 0.10,   # Wrong Item     → low risk
    2: 0.08,   # Transit Damage → low risk
    3: 0.85,   # False INR      → very high risk
    4: 0.60,   # Vague/Remorse  → elevated risk
}


# ──────────────────────────────────────────────────────────────
# DATA LOADING & SPLITTING
# ──────────────────────────────────────────────────────────────

def load_and_split(data_path: Path = DATA_PATH):
    """
    Load processed dataset and perform customer-grouped split
    70 / 15 / 15 using GroupShuffleSplit to prevent leakage.
    """
    print(f"Loading dataset from {data_path} …")
    df = pd.read_csv(data_path)
    print(f"  Total records: {len(df):,}")
    print(f"  Fraud rate   : {df['label'].mean():.1%}")

    groups = df["customer_unique_id"].values
    y      = df["label"].values

    # First split: train+val vs. test
    gss_outer = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=RANDOM_STATE)
    trainval_idx, test_idx = next(gss_outer.split(df, y, groups))

    df_trainval = df.iloc[trainval_idx].copy()
    df_test     = df.iloc[test_idx].copy()

    # Second split: train vs. val  (val = 15 / 85 ≈ 17.6% of trainval ≈ 15% total)
    gss_inner = GroupShuffleSplit(
        n_splits=1,
        test_size=0.15 / 0.85,
        random_state=RANDOM_STATE,
    )
    groups_tv = df_trainval["customer_unique_id"].values
    y_tv      = df_trainval["label"].values

    train_idx, val_idx = next(gss_inner.split(df_trainval, y_tv, groups_tv))

    df_train = df_trainval.iloc[train_idx].copy()
    df_val   = df_trainval.iloc[val_idx].copy()

    print(f"  Train: {len(df_train):,}  |  Val: {len(df_val):,}  |  Test: {len(df_test):,}")
    return df_train, df_val, df_test


# ──────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ──────────────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply additional feature engineering on top of simulated features.
    All transformations are deterministic.
    """
    df = df.copy()

    # Clip outliers
    df["days_to_return"]       = df["days_to_return"].clip(0, 60)
    df["account_age_days"]     = df["account_age_days"].clip(0, 3000)
    df["delivery_delay_days"]  = df["delivery_delay_days"].fillna(0).clip(0, 60)
    df["review_score"]         = df["review_score"].fillna(3.0).clip(1, 5)
    df["num_items"]            = df["num_items"].fillna(1).clip(1, 20)
    df["payment_installments"] = df["payment_installments"].fillna(1).clip(1, 24)
    df["total_price"]          = df["total_price"].fillna(0).clip(0, 10_000)

    # Log transforms for heavy-tailed numerics
    df["log_total_price"]       = np.log1p(df["total_price"])
    df["log_account_age"]       = np.log1p(df["account_age_days"])
    df["log_delivery_delay"]    = np.log1p(df["delivery_delay_days"])

    # Ratio features
    df["price_freight_ratio"]  = df["total_price"] / (df["total_freight"].fillna(1) + 1e-6)
    df["price_freight_ratio"]  = df["price_freight_ratio"].clip(0, 500)

    # Boolean combos
    df["high_ticket_new_account"] = (
        (df["is_high_ticket"] == 1) & (df["account_age_days"] < 30)
    ).astype(int)

    df["late_return_no_delay"] = (
        (df["days_to_return_ratio"] > 0.85) & (df["is_late_delivery_real"] == 0)
    ).astype(int)

    df["inr_carrier_confirmed"] = (
        (df.get("complaint_category", "") == "False INR") &
        (df["carrier_confirmed_delivery"] == 1)
    ).astype(int)

    df["serial_returner_flag"] = (df["customer_historical_return_rate"] > 0.35).astype(int)

    return df


def get_feature_columns(df_train: pd.DataFrame) -> list[str]:
    """Return the final list of tabular feature columns to use."""
    extra_engineered = [
        "log_total_price",
        "log_account_age",
        "log_delivery_delay",
        "price_freight_ratio",
        "high_ticket_new_account",
        "late_return_no_delay",
        "inr_carrier_confirmed",
        "serial_returner_flag",
    ]
    all_features = TABULAR_FEATURES + extra_engineered
    # Keep only columns that exist
    available = [f for f in all_features if f in df_train.columns]
    # Deduplicate preserving order
    seen = set()
    deduped = []
    for f in available:
        if f not in seen:
            seen.add(f)
            deduped.append(f)
    return deduped


# ──────────────────────────────────────────────────────────────
# LIGHTGBM TRAINING
# ──────────────────────────────────────────────────────────────

def train_lgbm(
    df_train: pd.DataFrame,
    df_val  : pd.DataFrame,
    feature_cols: list[str],
) -> lgb.LGBMClassifier:
    """
    Train LightGBM with cost-sensitive class weights.
    Uses early stopping on validation PR-AUC.
    """
    X_train = df_train[feature_cols].fillna(0).values
    y_train = df_train["label"].values
    X_val   = df_val[feature_cols].fillna(0).values
    y_val   = df_val["label"].values

    # Class weight: give fraud positive samples higher weight
    # proportional to (1 - fraud_rate) / fraud_rate  (balanced baseline)
    fraud_rate   = y_train.mean()
    scale_pos_wt = (1 - fraud_rate) / max(fraud_rate, 1e-6)
    # Moderate the weight (full balancing can over-penalise)
    scale_pos_wt = max(1.5, min(scale_pos_wt, 4.0))

    print(f"\n  Fraud rate         : {fraud_rate:.2%}")
    print(f"  scale_pos_weight   : {scale_pos_wt:.2f}")

    model = lgb.LGBMClassifier(
        n_estimators         = 600,
        learning_rate        = 0.03,
        num_leaves           = 63,
        max_depth            = -1,
        min_child_samples    = 20,
        subsample            = 0.80,
        colsample_bytree     = 0.80,
        reg_alpha            = 0.10,
        reg_lambda           = 0.20,
        scale_pos_weight     = scale_pos_wt,
        objective            = "binary",
        metric               = "average_precision",
        random_state         = RANDOM_STATE,
        n_jobs               = -1,
        verbose              = -1,
    )

    model.fit(
        X_train,
        y_train,
        eval_set              = [(X_val, y_val)],
        callbacks             = [
            lgb.early_stopping(stopping_rounds=50, verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    print(f"  Best iteration     : {model.best_iteration_}")
    return model


# ──────────────────────────────────────────────────────────────
# NLP PIPELINE TRAINING
# ──────────────────────────────────────────────────────────────

def train_nlp_pipeline(df_train: pd.DataFrame) -> Pipeline:
    """
    TF-IDF + Calibrated Logistic Regression for complaint text
    classification into 5 NLP categories.
    """
    texts  = df_train[NLP_TEXT_COL].fillna("no comment").tolist()
    labels = df_train[NLP_LABEL_COL].values

    tfidf = TfidfVectorizer(
        ngram_range  = (1, 2),
        max_features = 5_000,
        sublinear_tf = True,
        min_df       = 2,
    )

    base_lr = LogisticRegression(
        C           = 1.0,
        max_iter    = 500,
        solver      = "lbfgs",
        multi_class = "multinomial",
        random_state = RANDOM_STATE,
    )

    calibrated = CalibratedClassifierCV(base_lr, cv=3, method="isotonic")

    nlp_pipeline = Pipeline([
        ("tfidf",      tfidf),
        ("classifier", calibrated),
    ])

    nlp_pipeline.fit(texts, labels)
    print(f"  NLP pipeline trained on {len(texts):,} samples")
    return nlp_pipeline


# ──────────────────────────────────────────────────────────────
# SHAP EXPLAINER
# ──────────────────────────────────────────────────────────────

def build_shap_explainer(
    model       : lgb.LGBMClassifier,
    X_background: np.ndarray,
) -> shap.TreeExplainer:
    """Build and return a SHAP TreeExplainer for the LightGBM model."""
    explainer = shap.TreeExplainer(
        model,
        data                = shap.sample(X_background, 200, random_state=RANDOM_STATE),
        model_output        = "raw",
        feature_perturbation= "interventional",
    )
    return explainer


# ──────────────────────────────────────────────────────────────
# THRESHOLD OPTIMIZATION
# ──────────────────────────────────────────────────────────────

def find_cost_optimal_thresholds(
    y_true    : np.ndarray,
    y_prob    : np.ndarray,
    fp_cost   : int = FP_COST,
    fn_cost   : int = FN_COST,
    n_thresholds: int = 200,
) -> dict:
    """
    Grid-search two thresholds (T_low, T_high) that minimise:
        Cost = (fp_cost * N_FP) + (fn_cost * N_FN)

    Three-class rule:
        prob < T_low  → Genuine (Auto-approve)
        T_low ≤ prob < T_high → Unclear (Quick-check)
        prob ≥ T_high → Likely Fraud (Manual review)

    For cost calculation:
        FP = genuine classified as Fraud (T_high bucket)
        FN = fraud classified as Genuine (< T_low bucket)
    """
    thresholds = np.linspace(0.05, 0.95, n_thresholds)

    best_cost   = np.inf
    best_t_low  = 0.30
    best_t_high = 0.70

    for t_low in thresholds:
        for t_high in thresholds:
            if t_high <= t_low:
                continue

            pred_fraud = (y_prob >= t_high).astype(int)
            pred_genuine = (y_prob < t_low).astype(int)

            # FP: actually genuine but flagged as fraud
            fp = ((pred_fraud == 1) & (y_true == 0)).sum()
            # FN: actually fraud but cleared as genuine
            fn = ((pred_genuine == 1) & (y_true == 1)).sum()

            cost = fp_cost * fp + fn_cost * fn

            if cost < best_cost:
                best_cost   = cost
                best_t_low  = float(t_low)
                best_t_high = float(t_high)

    return {
        "t_low"     : round(best_t_low, 4),
        "t_high"    : round(best_t_high, 4),
        "total_cost": int(best_cost),
        "fp_cost"   : fp_cost,
        "fn_cost"   : fn_cost,
    }


# ──────────────────────────────────────────────────────────────
# EVALUATION
# ──────────────────────────────────────────────────────────────

def evaluate(
    model       : lgb.LGBMClassifier,
    nlp_pipeline: Pipeline,
    df_test     : pd.DataFrame,
    feature_cols: list[str],
    thresholds  : dict,
) -> str:
    """Full evaluation on test set. Returns a formatted report string."""
    X_test  = df_test[feature_cols].fillna(0).values
    y_test  = df_test["label"].values
    texts   = df_test[NLP_TEXT_COL].fillna("no comment").tolist()
    nlp_lbl = df_test[NLP_LABEL_COL].values

    # Behavioural model scores
    y_prob_bhv = model.predict_proba(X_test)[:, 1]

    # NLP text risk scores
    nlp_proba      = nlp_pipeline.predict_proba(texts)
    nlp_pred_class = np.argmax(nlp_proba, axis=1)
    nlp_risk       = np.array([NLP_RISK_MAP[c] for c in nlp_pred_class])

    # Composite score
    composite_prob = 0.65 * y_prob_bhv + 0.35 * nlp_risk

    t_low  = thresholds["t_low"]
    t_high = thresholds["t_high"]

    y_pred_bin = np.where(
        composite_prob >= t_high, 1,
        np.where(composite_prob < t_low, 0, -1)   # -1 = Unclear
    )

    # For binary metrics, treat Unclear as Fraud (conservative)
    y_pred_conservative = (composite_prob >= t_low).astype(int)
    y_pred_strict       = (composite_prob >= t_high).astype(int)

    pr_auc = average_precision_score(y_test, composite_prob)
    roc_auc = roc_auc_score(y_test, composite_prob)

    cm = confusion_matrix(y_test, y_pred_strict)
    fp = cm[0, 1] if cm.shape == (2, 2) else 0
    fn = cm[1, 0] if cm.shape == (2, 2) else 0
    business_cost = FP_COST * fp + FN_COST * fn

    lines = [
        "=" * 60,
        "  EVALUATION REPORT — Returns Fraud Sorter",
        "=" * 60,
        f"  Test set size    : {len(y_test):,}",
        f"  Fraud rate       : {y_test.mean():.2%}",
        "",
        "  ── Model Metrics (Composite Score) ─────────────────",
        f"  PR-AUC           : {pr_auc:.4f}",
        f"  ROC-AUC          : {roc_auc:.4f}",
        "",
        "  ── Threshold Configuration ─────────────────────────",
        f"  T_low            : {t_low:.4f}  (< T_low → Auto-approve)",
        f"  T_high           : {t_high:.4f}  (≥ T_high → Manual review)",
        "",
        "  ── Confusion Matrix (at T_high, strict) ────────────",
        f"  {cm}",
        "",
        f"  False Positives  : {fp}  (Genuine wrongly flagged)",
        f"  False Negatives  : {fn}  (Fraud wrongly cleared)",
        f"  Business Cost    : {business_cost:,}  (5×FP + 1×FN)",
        "",
        "  ── Classification Report ───────────────────────────",
        classification_report(y_test, y_pred_strict, target_names=["Genuine","Fraud"]),
        "",
        "  ── Triage Distribution ─────────────────────────────",
        f"  Auto-approve     : {(composite_prob < t_low).sum():>6,}",
        f"  Quick-check      : {((composite_prob >= t_low) & (composite_prob < t_high)).sum():>6,}",
        f"  Manual review    : {(composite_prob >= t_high).sum():>6,}",
        "=" * 60,
    ]

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# BASELINE HEURISTIC BENCHMARK
# ──────────────────────────────────────────────────────────────

def heuristic_baseline(df: pd.DataFrame) -> dict:
    """
    Simple rule-based heuristic for benchmark comparison.
    Flags a return as fraud if:
      - days_to_return_ratio > 0.85, OR
      - customer_historical_return_rate > 0.30, OR
      - (carrier_confirmed_delivery == 1 AND complaint is False INR)
    """
    y_true = df["label"].values

    heuristic_pred = (
        (df["days_to_return_ratio"] > 0.85) |
        (df["customer_historical_return_rate"] > 0.30) |
        (
            (df["carrier_confirmed_delivery"] == 1) &
            (df.get("complaint_category", "") == "False INR")
        )
    ).astype(int)

    heuristic_prob = heuristic_pred.astype(float) + np.random.default_rng(RANDOM_STATE).uniform(
        -0.1, 0.1, size=len(heuristic_pred)
    ).clip(-0.05, 0.05)
    heuristic_prob = heuristic_prob.clip(0, 1)

    pr_auc  = average_precision_score(y_true, heuristic_prob)
    roc_auc = roc_auc_score(y_true, heuristic_prob)
    cm      = confusion_matrix(y_true, heuristic_pred)

    return {
        "pr_auc"   : round(pr_auc, 4),
        "roc_auc"  : round(roc_auc, 4),
        "confusion": cm.tolist(),
    }


# ──────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Returns Fraud Sorter — Training Pipeline")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load & split
    print("\n[1/7] Loading and splitting data …")
    df_train, df_val, df_test = load_and_split()

    # 2. Feature engineering
    print("\n[2/7] Engineering features …")
    df_train = engineer_features(df_train)
    df_val   = engineer_features(df_val)
    df_test  = engineer_features(df_test)

    feature_cols = get_feature_columns(df_train)
    print(f"  Feature count: {len(feature_cols)}")

    # Save feature columns
    with open(MODELS_DIR / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2)

    # 3. Train LightGBM
    print("\n[3/7] Training LightGBM behavioural model …")
    lgbm_model = train_lgbm(df_train, df_val, feature_cols)

    X_val_arr  = df_val[feature_cols].fillna(0).values
    y_val_arr  = df_val["label"].values
    val_prob   = lgbm_model.predict_proba(X_val_arr)[:, 1]
    val_prauc  = average_precision_score(y_val_arr, val_prob)
    print(f"  Val PR-AUC: {val_prauc:.4f}")

    # 4. Train NLP pipeline
    print("\n[4/7] Training NLP complaint classifier …")
    nlp_pipeline = train_nlp_pipeline(df_train)

    # 5. SHAP explainer
    print("\n[5/7] Building SHAP TreeExplainer …")
    X_train_arr = df_train[feature_cols].fillna(0).values
    explainer   = build_shap_explainer(lgbm_model, X_train_arr)

    # 6. Threshold optimization on validation set
    print("\n[6/7] Optimizing cost-based thresholds on val set …")
    nlp_val_proba = nlp_pipeline.predict_proba(
        df_val[NLP_TEXT_COL].fillna("no comment").tolist()
    )
    nlp_val_class = np.argmax(nlp_val_proba, axis=1)
    nlp_val_risk  = np.array([NLP_RISK_MAP[c] for c in nlp_val_class])
    composite_val = 0.65 * val_prob + 0.35 * nlp_val_risk

    thresholds = find_cost_optimal_thresholds(y_val_arr, composite_val)
    print(f"  T_low  = {thresholds['t_low']}")
    print(f"  T_high = {thresholds['t_high']}")
    print(f"  Validation business cost = {thresholds['total_cost']:,}")

    with open(MODELS_DIR / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    # 7. Evaluation
    print("\n[7/7] Evaluating on test set …")
    report = evaluate(lgbm_model, nlp_pipeline, df_test, feature_cols, thresholds)
    print(report)

    # Heuristic baseline comparison
    print("\n  ── Heuristic Baseline Benchmark ────────────────────")
    heuristic_metrics = heuristic_baseline(df_test)
    print(f"  Heuristic PR-AUC  : {heuristic_metrics['pr_auc']}")
    print(f"  Heuristic ROC-AUC : {heuristic_metrics['roc_auc']}")
    print(f"  ML model PR-AUC   : (see report above)")

    report += (
        f"\n  ── Heuristic Baseline ───────────────────────────────\n"
        f"  Heuristic PR-AUC  : {heuristic_metrics['pr_auc']}\n"
        f"  Heuristic ROC-AUC : {heuristic_metrics['roc_auc']}\n"
    )

    # Save report
    report_path = MODELS_DIR / "training_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\n  Report saved → {report_path}")

    # Save models
    print("\n  Saving models …")
    joblib.dump(lgbm_model,    MODELS_DIR / "lgbm_model.pkl")
    joblib.dump(nlp_pipeline,  MODELS_DIR / "nlp_pipeline.pkl")
    joblib.dump(explainer,     MODELS_DIR / "shap_explainer.pkl")

    # Save feature names for the explainer
    feature_meta = {
        "feature_columns" : feature_cols,
        "nlp_label_names" : NLP_LABEL_NAMES,
        "nlp_risk_map"    : NLP_RISK_MAP,
    }
    with open(MODELS_DIR / "feature_meta.json", "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2)

    print("\n✓ Training complete. Models saved to models/")
    print("  Run `python app.py` to launch the Streamlit dashboard.")


if __name__ == "__main__":
    main()
