"""
pipeline.py
────────────────────────────────────────────────────────────────
Single-class inference engine for the Returns Fraud vs. Genuine
Grievance Sorter.

The `ReturnSorterPipeline` class:
  - Loads trained LightGBM, NLP, SHAP models from disk (lazy)
  - Accepts an order + return details dict
  - Runs both models
  - Extracts top SHAP-based plain-English risk drivers
  - Returns a structured TriageResult with all 4 outputs

Usage:
    from src.pipeline import ReturnSorterPipeline

    pipeline = ReturnSorterPipeline()
    result   = pipeline.predict(order_dict)
    print(result)
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import shap

# ──────────────────────────────────────────────────────────────
# CONSTANTS & PATHS
# ──────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")

# Composite weighting
BEHAVIOR_WEIGHT = 0.65
NLP_WEIGHT      = 0.35

# NLP risk map (fraud probability proxy per complaint class)
NLP_RISK_MAP = {
    0: 0.05,   # Defective
    1: 0.10,   # Wrong Item
    2: 0.08,   # Transit Damage
    3: 0.85,   # False INR
    4: 0.60,   # Vague/Remorse
}

NLP_LABEL_NAMES = {
    0: "Defective",
    1: "Wrong Item",
    2: "Transit Damage",
    3: "False INR",
    4: "Vague/Remorse",
}

# Human-readable feature name mappings for SHAP explanations
FEATURE_DISPLAY_NAMES = {
    "days_to_return"                    : "Days until return was filed",
    "days_to_return_ratio"              : "Return timing ratio (vs. window)",
    "freight_ratio"                     : "Freight cost as % of order value",
    "is_late_delivery_real"             : "Confirmed carrier delay",
    "price_vs_category_median"          : "Price vs. category median",
    "customer_historical_return_rate"   : "Customer's past return rate",
    "product_category_return_propensity": "Category return propensity",
    "payment_installments"              : "Number of payment installments",
    "carrier_confirmed_delivery"        : "Carrier confirmed delivery",
    "is_high_ticket"                    : "High-ticket item",
    "is_fit_sensitive"                  : "Fit-sensitive category (fashion)",
    "is_fragile_category"               : "Fragile/heavy item category",
    "account_age_days"                  : "Customer account age (days)",
    "delivery_delay_days"               : "Actual delivery delay (days)",
    "review_score"                      : "Customer review rating",
    "num_items"                         : "Number of items in order",
    "total_price"                       : "Total order value (BRL)",
    "log_total_price"                   : "Log of total order value",
    "log_account_age"                   : "Log of account age",
    "log_delivery_delay"                : "Log of delivery delay",
    "price_freight_ratio"               : "Price-to-freight ratio",
    "high_ticket_new_account"           : "High-value order from new account",
    "late_return_no_delay"              : "Late return without real delay",
    "inr_carrier_confirmed"             : "INR claim with confirmed delivery",
    "serial_returner_flag"              : "Serial returner flag",
}

# Human-readable risk explanations
SHAP_RISK_REASONS = {
    "days_to_return_ratio"           : ("↑ Return filed very close to window deadline",
                                        "↓ Return filed promptly after issue"),
    "customer_historical_return_rate": ("↑ Customer has high historical return rate",
                                        "↓ Customer has low return history"),
    "carrier_confirmed_delivery"     : ("↑ Carrier confirmed delivery despite INR claim",
                                        "↓ Delivery status unconfirmed"),
    "high_ticket_new_account"        : ("↑ High-value order from a brand-new account",
                                        "↓ Established account ordering normally"),
    "late_return_no_delay"           : ("↑ Return filed late but no real delivery delay",
                                        "↓ Return timing aligns with delivery issues"),
    "inr_carrier_confirmed"          : ("↑ Claimed non-receipt but delivery confirmed",
                                        "↓ Non-receipt claim without delivery confirmation"),
    "serial_returner_flag"           : ("↑ Customer flagged as serial returner",
                                        "↓ Not a frequent returner"),
    "price_vs_category_median"       : ("↑ Item priced significantly above category norm",
                                        "↓ Item priced within normal range"),
    "is_high_ticket"                 : ("↑ Expensive/luxury item involved",
                                        "↓ Standard-priced item"),
    "account_age_days"               : ("↑ Very new customer account",
                                        "↓ Long-standing customer account"),
    "review_score"                   : ("↑ Review score inconsistent with complaint",
                                        "↓ Low review score supports genuine grievance"),
    "payment_installments"           : ("↑ Many installments on a refund request",
                                        "↓ Normal payment terms"),
    "is_late_delivery_real"          : ("↓ Real carrier delay supports the complaint",
                                        "↑ No actual delivery delay recorded"),
    "delivery_delay_days"            : ("↓ Significant real delay supports the claim",
                                        "↑ No meaningful delay, claim suspicious"),
}


# ──────────────────────────────────────────────────────────────
# TRIAGE RESULT DATACLASS
# ──────────────────────────────────────────────────────────────

@dataclass
class TriageResult:
    """Structured output from the ReturnSorterPipeline."""

    # (a) Risk score 0-100
    risk_score: float

    # (b) Triage category
    category: str   # "Likely Fraud" | "Genuine Grievance" | "Unclear"

    # (c) Top SHAP reasons (plain English)
    top_reasons: list[str]

    # (d) Recommended action
    recommended_action: str   # "Auto-approve" | "Quick check" | "Manual review"

    # Supplementary details
    behavior_prob: float         = 0.0
    nlp_risk_score: float        = 0.0
    nlp_complaint_category: str  = "Unknown"
    shap_values: list[float]     = field(default_factory=list)
    shap_feature_names: list[str]= field(default_factory=list)
    t_low: float                 = 0.30
    t_high: float                = 0.70

    def __str__(self) -> str:
        lines = [
            "─" * 50,
            f"  TRIAGE RESULT",
            "─" * 50,
            f"  Risk Score        : {self.risk_score:.1f} / 100",
            f"  Category          : {self.category}",
            f"  Action            : {self.recommended_action}",
            f"  Behavior Prob     : {self.behavior_prob:.4f}",
            f"  NLP Category      : {self.nlp_complaint_category}",
            f"  NLP Risk Score    : {self.nlp_risk_score:.4f}",
            "",
            "  Top Risk Drivers:",
        ]
        for i, reason in enumerate(self.top_reasons, 1):
            lines.append(f"    {i}. {reason}")
        lines.append("─" * 50)
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "risk_score"           : round(self.risk_score, 2),
            "category"             : self.category,
            "top_reasons"          : self.top_reasons,
            "recommended_action"   : self.recommended_action,
            "behavior_prob"        : round(self.behavior_prob, 4),
            "nlp_risk_score"       : round(self.nlp_risk_score, 4),
            "nlp_complaint_category": self.nlp_complaint_category,
        }


# ──────────────────────────────────────────────────────────────
# FEATURE ENGINEERING (mirrors train.py)
# ──────────────────────────────────────────────────────────────

def _engineer_single_record(record: dict) -> dict:
    """
    Apply the same feature engineering transformations as train.py
    but for a single inference record (dict → dict).
    """
    r = dict(record)  # Copy to avoid mutation

    # Safe float/int conversion helpers
    def _f(key: str, default: float = 0.0) -> float:
        try:
            return float(r.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    def _i(key: str, default: int = 0) -> int:
        try:
            return int(r.get(key, default) or default)
        except (TypeError, ValueError):
            return default

    # Raw fields
    total_price   = _f("total_price",    100.0)
    total_freight = _f("total_freight",  10.0)
    days_return   = _f("days_to_return", 7.0)
    return_window = _f("return_window_days", 30.0)
    acct_age      = _f("account_age_days", 365.0)
    delay_days    = _f("delivery_delay_days", 0.0)
    cat_median    = _f("cat_median_price", 100.0)

    # Derived
    r["freight_ratio"]           = (total_freight / (total_price + 1e-6))
    r["days_to_return_ratio"]    = days_return / (return_window + 1e-6)
    r["price_vs_category_median"]= total_price / (cat_median + 1e-6)
    r["log_total_price"]         = np.log1p(total_price)
    r["log_account_age"]         = np.log1p(acct_age)
    r["log_delivery_delay"]      = np.log1p(delay_days)
    r["price_freight_ratio"]     = total_price / (total_freight + 1e-6)

    r["high_ticket_new_account"] = int(
        _i("is_high_ticket") == 1 and acct_age < 30
    )
    r["late_return_no_delay"] = int(
        r["days_to_return_ratio"] > 0.85 and _i("is_late_delivery_real") == 0
    )
    r["inr_carrier_confirmed"] = int(
        str(r.get("complaint_category", "")).lower() == "false inr" and
        _i("carrier_confirmed_delivery") == 1
    )
    r["serial_returner_flag"] = int(
        _f("customer_historical_return_rate") > 0.35
    )

    return r


# ──────────────────────────────────────────────────────────────
# PIPELINE CLASS
# ──────────────────────────────────────────────────────────────

class ReturnSorterPipeline:
    """
    Inference pipeline for return triage.

    Attributes
    ----------
    models_dir : Path
        Directory containing serialized model artifacts.
    _lgbm      : lgb.LGBMClassifier (lazy-loaded)
    _nlp       : sklearn Pipeline   (lazy-loaded)
    _explainer : shap.TreeExplainer (lazy-loaded)
    _features  : list[str]          (lazy-loaded)
    _thresholds: dict               (lazy-loaded)
    """

    def __init__(self, models_dir: Path | str = MODELS_DIR):
        self.models_dir  = Path(models_dir)
        self._lgbm       = None
        self._nlp        = None
        self._explainer  = None
        self._features   = None
        self._thresholds = None

    # ── Lazy Loaders ──────────────────────────────────────────

    def _load_models(self):
        """Load all model artifacts from disk (called once)."""
        if self._lgbm is not None:
            return  # Already loaded

        lgbm_path      = self.models_dir / "lgbm_model.pkl"
        nlp_path       = self.models_dir / "nlp_pipeline.pkl"
        explainer_path = self.models_dir / "shap_explainer.pkl"
        features_path  = self.models_dir / "feature_columns.json"
        thresh_path    = self.models_dir / "thresholds.json"

        for p in [lgbm_path, nlp_path, explainer_path, features_path, thresh_path]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Model artifact not found: {p}\n"
                    "Run `python src/train.py` first."
                )

        self._lgbm      = joblib.load(lgbm_path)
        self._nlp       = joblib.load(nlp_path)
        self._explainer = joblib.load(explainer_path)

        with open(features_path) as f:
            self._features = json.load(f)

        with open(thresh_path) as f:
            self._thresholds = json.load(f)

    # ── Core Prediction ───────────────────────────────────────

    def predict(
        self,
        order: dict[str, Any],
        fp_cost: int | None = None,
        fn_cost: int | None = None,
    ) -> TriageResult:
        """
        Run the full triage pipeline on a single return request.

        Parameters
        ----------
        order : dict
            Must contain at minimum:
              - complaint_text (str)
              - days_to_return (float)
            Optional keys (all default to safe values if missing):
              - total_price, total_freight, freight_ratio
              - is_late_delivery_real, delivery_delay_days
              - price_vs_category_median, cat_median_price
              - customer_historical_return_rate
              - product_category_return_propensity
              - payment_installments
              - carrier_confirmed_delivery
              - is_high_ticket, is_fit_sensitive, is_fragile_category
              - account_age_days, review_score, num_items
              - return_window_days, complaint_category
        fp_cost : int, optional
            Override FP cost weight for dynamic threshold re-computation.
        fn_cost : int, optional
            Override FN cost weight.

        Returns
        -------
        TriageResult
        """
        self._load_models()

        # ── Step 1: Feature engineering ──────────────────────
        engineered = _engineer_single_record(order)

        # Build feature vector
        x_dict = {}
        for feat in self._features:
            val = engineered.get(feat, 0)
            try:
                x_dict[feat] = float(val) if val is not None else 0.0
            except (TypeError, ValueError):
                x_dict[feat] = 0.0

        x_arr = np.array([[x_dict[f] for f in self._features]], dtype=np.float32)

        # ── Step 2: Behavioural model ─────────────────────────
        behavior_prob = float(self._lgbm.predict_proba(x_arr)[0, 1])

        # ── Step 3: NLP model ─────────────────────────────────
        complaint_text = str(order.get("complaint_text", "no comment"))
        nlp_proba      = self._nlp.predict_proba([complaint_text])[0]
        nlp_class      = int(np.argmax(nlp_proba))
        nlp_risk       = NLP_RISK_MAP[nlp_class]
        nlp_category   = NLP_LABEL_NAMES[nlp_class]

        # ── Step 4: Composite score ───────────────────────────
        composite = BEHAVIOR_WEIGHT * behavior_prob + NLP_WEIGHT * nlp_risk
        risk_score = float(composite * 100)

        # ── Step 5: Thresholds ────────────────────────────────
        if fp_cost is not None and fn_cost is not None:
            # Recompute thresholds at runtime from stored val data
            # (use stored thresholds scaled by cost ratio)
            base_fp   = self._thresholds.get("fp_cost", 5)
            base_fn   = self._thresholds.get("fn_cost", 1)
            base_ratio = base_fp / max(base_fn, 1e-6)
            new_ratio  = fp_cost / max(fn_cost, 1e-6)
            scaling    = base_ratio / max(new_ratio, 1e-6)
            # Shift thresholds: higher FP cost → lower thresholds (approve more)
            t_low  = float(np.clip(self._thresholds["t_low"]  * scaling, 0.05, 0.50))
            t_high = float(np.clip(self._thresholds["t_high"] * scaling, 0.30, 0.90))
            t_high = max(t_high, t_low + 0.10)
        else:
            t_low  = self._thresholds["t_low"]
            t_high = self._thresholds["t_high"]

        # Normalise composite to [0, 1] for threshold comparison
        composite_norm = composite

        # ── Step 6: Category & action ─────────────────────────
        if composite_norm >= t_high:
            category           = "Likely Fraud"
            recommended_action = "Manual review"
        elif composite_norm < t_low:
            category           = "Genuine Grievance"
            recommended_action = "Auto-approve"
        else:
            category           = "Unclear"
            recommended_action = "Quick check"

        # ── Step 7: SHAP explanation ──────────────────────────
        shap_values, top_reasons = self._compute_shap_reasons(x_arr, behavior_prob)

        return TriageResult(
            risk_score             = round(risk_score, 2),
            category               = category,
            top_reasons            = top_reasons,
            recommended_action     = recommended_action,
            behavior_prob          = round(behavior_prob, 4),
            nlp_risk_score         = round(nlp_risk, 4),
            nlp_complaint_category = nlp_category,
            shap_values            = shap_values,
            shap_feature_names     = self._features,
            t_low                  = t_low,
            t_high                 = t_high,
        )

    def _compute_shap_reasons(
        self,
        x_arr       : np.ndarray,
        behavior_prob: float,
    ) -> tuple[list[float], list[str]]:
        """
        Compute SHAP values for the single input and map to
        plain-English explanations.

        Returns
        -------
        shap_values : list[float]   Raw SHAP values per feature
        top_reasons : list[str]     Top 5 plain-English risk drivers
        """
        try:
            sv = self._explainer.shap_values(x_arr)

            # Handle list output (multi-output / SHAP versions differ)
            if isinstance(sv, list):
                sv = sv[1] if len(sv) > 1 else sv[0]

            shap_row = sv[0]   # Shape: (n_features,)

        except Exception:
            # Fallback: uniform importance if SHAP fails
            shap_row = np.zeros(len(self._features))

        shap_values = [float(v) for v in shap_row]

        # Sort by absolute SHAP value (descending)
        sorted_idx = np.argsort(np.abs(shap_row))[::-1]

        top_reasons = []
        for idx in sorted_idx[:5]:
            feat      = self._features[idx]
            sv_val    = float(shap_row[idx])
            is_risky  = sv_val > 0   # Positive SHAP → pushes toward fraud

            if feat in SHAP_RISK_REASONS:
                reason_tuple = SHAP_RISK_REASONS[feat]
                reason = reason_tuple[0] if is_risky else reason_tuple[1]
            else:
                display = FEATURE_DISPLAY_NAMES.get(feat, feat)
                direction = "increases" if is_risky else "decreases"
                reason = f"{display} {direction} risk (SHAP={sv_val:+.3f})"

            top_reasons.append(reason)

        return shap_values, top_reasons

    # ── Batch Prediction ──────────────────────────────────────

    def predict_batch(
        self,
        orders: list[dict[str, Any]],
        fp_cost: int | None = None,
        fn_cost: int | None = None,
    ) -> list[TriageResult]:
        """Run predict() over a list of orders."""
        return [self.predict(o, fp_cost=fp_cost, fn_cost=fn_cost) for o in orders]


# ──────────────────────────────────────────────────────────────
# PRESET SCENARIOS (used by Streamlit app)
# ──────────────────────────────────────────────────────────────

PRESET_SCENARIOS = {
    "Wardrobing": {
        "label"              : "🕶️ Wardrobing Scenario",
        "description"        : "Customer returns a luxury fashion item on day 28 of a 30-day window.",
        "days_to_return"     : 28,
        "return_window_days" : 30,
        "total_price"        : 589.90,
        "total_freight"      : 22.50,
        "cat_median_price"   : 380.00,
        "complaint_text"     : "Changed my mind, no longer need it",
        "complaint_category" : "Vague/Remorse",
        "is_late_delivery_real"           : 0,
        "delivery_delay_days"             : 0,
        "carrier_confirmed_delivery"      : 1,
        "is_high_ticket"                  : 1,
        "is_fit_sensitive"                : 1,
        "is_fragile_category"             : 0,
        "account_age_days"                : 420,
        "customer_historical_return_rate" : 0.08,
        "product_category_return_propensity": 0.28,
        "payment_installments"            : 6,
        "review_score"                    : 4.0,
        "num_items"                       : 1,
    },
    "Transit Delay": {
        "label"              : "📦 Transit Delay Scenario",
        "description"        : "Package arrived 9 days late; customer complains of damage.",
        "days_to_return"     : 4,
        "return_window_days" : 30,
        "total_price"        : 189.90,
        "total_freight"      : 18.00,
        "cat_median_price"   : 175.00,
        "complaint_text"     : "Package arrived completely crushed and item is broken",
        "complaint_category" : "Transit Damage",
        "is_late_delivery_real"           : 1,
        "delivery_delay_days"             : 9.0,
        "carrier_confirmed_delivery"      : 1,
        "is_high_ticket"                  : 0,
        "is_fit_sensitive"                : 0,
        "is_fragile_category"             : 1,
        "account_age_days"                : 730,
        "customer_historical_return_rate" : 0.04,
        "product_category_return_propensity": 0.15,
        "payment_installments"            : 1,
        "review_score"                    : 1.0,
        "num_items"                       : 1,
    },
    "False INR": {
        "label"              : "🕵️ False INR Scenario",
        "description"        : "Buyer claims non-receipt but carrier tracking confirms delivery.",
        "days_to_return"     : 2,
        "return_window_days" : 30,
        "total_price"        : 450.00,
        "total_freight"      : 15.00,
        "cat_median_price"   : 300.00,
        "complaint_text"     : "I never received this order. Tracking says delivered but nothing arrived at my address.",
        "complaint_category" : "False INR",
        "is_late_delivery_real"           : 0,
        "delivery_delay_days"             : 0,
        "carrier_confirmed_delivery"      : 1,
        "is_high_ticket"                  : 1,
        "is_fit_sensitive"                : 0,
        "is_fragile_category"             : 0,
        "account_age_days"                : 8,
        "customer_historical_return_rate" : 0.55,
        "product_category_return_propensity": 0.18,
        "payment_installments"            : 1,
        "review_score"                    : 3.0,
        "num_items"                       : 1,
    },
    "Clear Defect": {
        "label"              : "🔧 Clear Defect Scenario",
        "description"        : "Product defective from day 1; 1-star review with defect complaint.",
        "days_to_return"     : 3,
        "return_window_days" : 30,
        "total_price"        : 120.00,
        "total_freight"      : 14.00,
        "cat_median_price"   : 130.00,
        "complaint_text"     : "Product arrived broken and unusable. Stopped working after first use.",
        "complaint_category" : "Defective",
        "is_late_delivery_real"           : 0,
        "delivery_delay_days"             : 0,
        "carrier_confirmed_delivery"      : 1,
        "is_high_ticket"                  : 0,
        "is_fit_sensitive"                : 0,
        "is_fragile_category"             : 0,
        "account_age_days"                : 900,
        "customer_historical_return_rate" : 0.03,
        "product_category_return_propensity": 0.10,
        "payment_installments"            : 1,
        "review_score"                    : 1.0,
        "num_items"                       : 1,
    },
}


# ──────────────────────────────────────────────────────────────
# STANDALONE TEST
# ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    pipeline = ReturnSorterPipeline()

    print("\nRunning preset scenarios …\n")
    for name, scenario in PRESET_SCENARIOS.items():
        result = pipeline.predict(scenario)
        print(f"── {name} ──")
        print(result)
        print()
