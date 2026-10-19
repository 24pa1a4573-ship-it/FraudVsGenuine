"""
simulate_returns.py
────────────────────────────────────────────────────────────────
Synthetic Return Generator for the Returns Fraud vs. Genuine
Grievance Sorter system.

Generates realistic return records on top of Olist transaction
data, using four fraud archetypes and three genuine archetypes,
with 10-15% label noise injected for realism.

Usage:
    python src/simulate_returns.py
Outputs:
    data/processed/returns_dataset.csv
    data/processed/returns_dataset_meta.json
"""

import os
import json
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

warnings.filterwarnings("ignore")


def _safe_int(val, default: int = 0) -> int:
    """Convert val to int, returning *default* if val is NaN or None."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return int(val)
    except (TypeError, ValueError):
        return default


def _safe_float(val, default: float = 0.0) -> float:
    """Convert val to float, returning *default* if val is NaN or None."""
    try:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return default
        return float(val)
    except (TypeError, ValueError):
        return default

# ──────────────────────────────────────────────────────────────
# CONSTANTS & PATHS
# ──────────────────────────────────────────────────────────────
RANDOM_STATE = 42
RETURN_RATE  = 0.175          # 17.5 % of orders generate a return
FRAUD_RATIO  = 0.35           # 35 % of returns are fraudulent
NOISE_RATE   = 0.12           # 12 % label noise injected

DATA_DIR        = Path("data")
RAW_DIR         = DATA_DIR / "raw"
PROCESSED_DIR   = DATA_DIR / "processed"

OLIST_FILES = {
    "orders"    : RAW_DIR / "olist_orders_dataset.csv",
    "items"     : RAW_DIR / "olist_order_items_dataset.csv",
    "reviews"   : RAW_DIR / "olist_order_reviews_dataset.csv",
    "payments"  : RAW_DIR / "olist_order_payments_dataset.csv",
    "products"  : RAW_DIR / "olist_products_dataset.csv",
    "customers" : RAW_DIR / "olist_customers_dataset.csv",
}

# ──────────────────────────────────────────────────────────────
# CATEGORY CONFIGURATION
# ──────────────────────────────────────────────────────────────
HIGH_TICKET_CATEGORIES = {
    "watches_gifts", "computers", "consoles_games",
    "telephony", "computers_accessories", "electronics",
    "perfumery", "fashion_bags_accessories",
}

FIT_SENSITIVE_CATEGORIES = {
    "fashion_male_clothing", "fashion_female_clothing",
    "fashion_childrens_clothes", "fashion_shoes",
    "fashion_sport", "fashion_underwear_beach",
    "fashion_bags_accessories",
}

FRAGILE_CATEGORIES = {
    "home_appliances", "home_appliances_2", "small_appliances",
    "kitchen_dining_laundry_garden_furniture", "furniture_bedroom",
    "furniture_living_room", "furniture_mattress_and_upholstery",
    "art", "musical_instruments",
}

COMPLAINT_TEMPLATES = {
    "Defective": [
        "Product arrived broken and unusable",
        "Item stopped working after first use",
        "Received a defective unit, not as described",
        "The product is faulty and does not turn on",
        "Quality is terrible, broke immediately",
        "Item has manufacturing defects, requesting replacement",
        "Product malfunction from day one",
    ],
    "Wrong Item": [
        "Received wrong item, not what I ordered",
        "Sent the completely different product",
        "Color and size do not match my order",
        "Wrong model delivered to me",
        "Package contained an entirely different product",
        "Order mix-up, received someone else's item",
    ],
    "Transit Damage": [
        "Package arrived completely crushed",
        "Item was damaged during shipping",
        "Box was torn open and product inside broken",
        "Delivery caused the fragile item to shatter",
        "Serious damage in transit, unusable condition",
        "Poor packaging led to item being destroyed",
    ],
    "False INR": [
        "I never received this order",
        "Package was not delivered to my address",
        "Tracking says delivered but nothing arrived",
        "I was home all day and no delivery came",
        "Stolen from doorstep, need refund",
        "Item not in my possession despite tracking update",
    ],
    "Vague/Remorse": [
        "Changed my mind, no longer need it",
        "Doesn't fit my needs",
        "Bought by mistake",
        "Not satisfied with the quality",
        "Decided I don't want it anymore",
        "Product is okay but just not for me",
        "Returning because I found a better deal",
        "Did not like it",
    ],
}

NLP_LABEL_MAP = {
    "Defective"      : 0,
    "Wrong Item"     : 1,
    "Transit Damage" : 2,
    "False INR"      : 3,
    "Vague/Remorse"  : 4,
}

# ──────────────────────────────────────────────────────────────
# DATA LOADING HELPERS
# ──────────────────────────────────────────────────────────────

def _load_olist() -> dict[str, pd.DataFrame]:
    """Load all Olist CSV files; raise FileNotFoundError if missing."""
    frames = {}
    for key, path in OLIST_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"Missing Olist file: {path}\n"
                "Download from https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce "
                f"and place CSVs in {RAW_DIR}/"
            )
        frames[key] = pd.read_csv(path)
        print(f"  Loaded {key:<12} → {len(frames[key]):>7,} rows")
    return frames


def _build_base_frame(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Join orders → items → reviews → payments → products → customers
    into a single denormalized transaction frame.
    """
    orders    = frames["orders"].copy()
    items     = frames["items"].copy()
    reviews   = frames["reviews"].copy()
    payments  = frames["payments"].copy()
    products  = frames["products"].copy()
    customers = frames["customers"].copy()

    # Aggregate items per order
    items_agg = (
        items.groupby("order_id")
        .agg(
            num_items          = ("order_item_id", "count"),
            total_price        = ("price", "sum"),
            total_freight      = ("freight_value", "sum"),
            max_item_price     = ("price", "max"),
            product_id_first   = ("product_id", "first"),
            seller_id          = ("seller_id", "first"),
        )
        .reset_index()
    )

    # Payment aggregation
    pay_agg = (
        payments.groupby("order_id")
        .agg(
            payment_value        = ("payment_value", "sum"),
            payment_installments = ("payment_installments", "max"),
            payment_type         = ("payment_type", "first"),
        )
        .reset_index()
    )

    # Best review per order (latest)
    rev_agg = (
        reviews.sort_values("review_answer_timestamp")
        .groupby("order_id")
        .agg(
            review_score   = ("review_score", "last"),
            review_comment = ("review_comment_message", "last"),
        )
        .reset_index()
    )

    # Product category
    prod_cat = products[["product_id", "product_category_name"]].drop_duplicates()

    # Parse dates
    date_cols = [
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    ]
    for col in date_cols:
        if col in orders.columns:
            orders[col] = pd.to_datetime(orders[col], errors="coerce")

    # Merge chain
    df = (
        orders
        .merge(items_agg,  on="order_id",   how="left")
        .merge(pay_agg,    on="order_id",   how="left")
        .merge(rev_agg,    on="order_id",   how="left")
        .merge(customers,  on="customer_id", how="left")
        .merge(prod_cat,   left_on="product_id_first", right_on="product_id", how="left")
    )

    # Only keep delivered orders with enough data
    df = df[df["order_status"] == "delivered"].copy()
    df = df.dropna(subset=["order_delivered_customer_date", "total_price"])

    # Derived columns
    df["is_late_delivery_real"] = (
        df["order_delivered_customer_date"] > df["order_estimated_delivery_date"]
    ).astype(int)

    df["delivery_delay_days"] = (
        (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"])
        .dt.total_seconds() / 86_400
    ).clip(lower=0)

    df["freight_ratio"] = (
        df["total_freight"] / (df["total_price"] + 1e-6)
    ).clip(0, 5)

    df["product_category_name"] = (
        df["product_category_name"].fillna("unknown").str.lower().str.strip()
    )

    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# CATEGORY STATISTICS (price medians, return propensity)
# ──────────────────────────────────────────────────────────────

def _compute_category_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-category median price and synthetic return propensity.
    Higher-ticket and fit-sensitive categories have higher propensity.
    """
    cat_stats = (
        df.groupby("product_category_name")
        .agg(
            cat_median_price    = ("total_price", "median"),
            cat_order_count     = ("order_id",    "count"),
        )
        .reset_index()
    )

    # Synthetic return propensity signal (0–1)
    def _propensity(cat: str) -> float:
        cat = cat.lower()
        if any(c in cat for c in ["fashion", "clothing", "shoes", "sport"]):
            return 0.28
        if any(c in cat for c in ["computer", "telephony", "watches", "perfume"]):
            return 0.18
        if any(c in cat for c in ["furniture", "home_appliance", "kitchen"]):
            return 0.15
        return 0.10

    cat_stats["product_category_return_propensity"] = (
        cat_stats["product_category_name"].apply(_propensity)
    )

    return cat_stats


# ──────────────────────────────────────────────────────────────
# CUSTOMER HISTORY (historical return rate)
# ──────────────────────────────────────────────────────────────

def _compute_customer_history(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Simulate per-customer historical return rate.
    Serial-returner clusters get injected 0.4–0.9 rates.
    """
    customers = df["customer_unique_id"].unique()
    n = len(customers)

    # Base rate: log-normal centred around 5 %
    base_rate = rng.lognormal(mean=-3.0, sigma=1.0, size=n).clip(0, 1)

    # Serial-returner cluster (~3 % of customers)
    serial_mask = rng.random(n) < 0.03
    base_rate[serial_mask] = rng.uniform(0.40, 0.90, size=serial_mask.sum())

    hist_df = pd.DataFrame({
        "customer_unique_id"          : customers,
        "customer_historical_return_rate": base_rate,
    })
    return hist_df


# ──────────────────────────────────────────────────────────────
# ARCHETYPE GENERATORS
# ──────────────────────────────────────────────────────────────

def _wardrobing_archetype(row: pd.Series, rng: np.random.Generator) -> dict:
    """Returns initiated 27–29 days into a 30-day window on luxury items."""
    days = rng.integers(27, 30)
    return {
        "days_to_return"     : int(days),
        "return_window_days" : 30,
        "complaint_category" : "Vague/Remorse",
        "complaint_text"     : rng.choice(COMPLAINT_TEMPLATES["Vague/Remorse"]),
        "is_high_ticket"     : 1,
        "archetype"          : "Wardrobing",
        "label"              : 1,  # Fraud
    }


def _false_inr_archetype(row: pd.Series, rng: np.random.Generator) -> dict:
    """Buyer claims non-receipt but delivery was confirmed."""
    days = rng.integers(1, 5)
    return {
        "days_to_return"     : int(days),
        "return_window_days" : 30,
        "complaint_category" : "False INR",
        "complaint_text"     : rng.choice(COMPLAINT_TEMPLATES["False INR"]),
        "carrier_confirmed_delivery": 1,
        "is_high_ticket"     : int(row.get("total_price", 100) > 150),
        "archetype"          : "False INR",
        "label"              : 1,
    }


def _serial_returner_archetype(row: pd.Series, rng: np.random.Generator) -> dict:
    """Burst of returns from high-churn customer cluster."""
    days = rng.integers(3, 20)
    cat  = rng.choice(list(COMPLAINT_TEMPLATES.keys()))
    return {
        "days_to_return"     : int(days),
        "return_window_days" : 30,
        "complaint_category" : cat,
        "complaint_text"     : rng.choice(COMPLAINT_TEMPLATES[cat]),
        "is_high_ticket"     : int(row.get("total_price", 100) > 100),
        "archetype"          : "Serial Returner",
        "label"              : 1,
    }


def _new_account_arbitrage_archetype(row: pd.Series, rng: np.random.Generator) -> dict:
    """Fresh account, high-value item, instant refund request."""
    days = rng.integers(0, 3)
    return {
        "days_to_return"     : int(days),
        "return_window_days" : 30,
        "complaint_category" : "Vague/Remorse",
        "complaint_text"     : rng.choice(COMPLAINT_TEMPLATES["Vague/Remorse"]),
        "is_high_ticket"     : 1,
        "account_age_days"   : rng.integers(0, 14),
        "archetype"          : "New Account Arbitrage",
        "label"              : 1,
    }


# ──────────────────────────────────────────────────────────────
# GENUINE ARCHETYPE GENERATORS
# ──────────────────────────────────────────────────────────────

def _real_delay_archetype(row: pd.Series, rng: np.random.Generator) -> dict:
    """Delivered late; customer files return due to frustration."""
    days = rng.integers(2, 15)
    return {
        "days_to_return"          : int(days),
        "return_window_days"      : 30,
        "complaint_category"      : "Transit Damage",
        "complaint_text"          : rng.choice(COMPLAINT_TEMPLATES["Transit Damage"]),
        "is_late_delivery_real"   : 1,
        "delivery_delay_days"     : float(rng.uniform(3, 20)),
        "archetype"               : "Real Delay",
        "label"                   : 0,  # Genuine
    }


def _defective_item_archetype(row: pd.Series, rng: np.random.Generator) -> dict:
    """Low-star review with defect complaint."""
    days = rng.integers(1, 10)
    return {
        "days_to_return"     : int(days),
        "return_window_days" : 30,
        "complaint_category" : "Defective",
        "complaint_text"     : rng.choice(COMPLAINT_TEMPLATES["Defective"]),
        "review_score"       : rng.choice([1, 2]),
        "archetype"          : "Defective Item",
        "label"              : 0,
    }


def _size_mismatch_archetype(row: pd.Series, rng: np.random.Generator) -> dict:
    """Fit-sensitive or fragile category mismatch."""
    days = rng.integers(2, 12)
    return {
        "days_to_return"     : int(days),
        "return_window_days" : 30,
        "complaint_category" : "Wrong Item",
        "complaint_text"     : rng.choice(COMPLAINT_TEMPLATES["Wrong Item"]),
        "is_fit_sensitive"   : 1,
        "archetype"          : "Size/Category Mismatch",
        "label"              : 0,
    }


# ──────────────────────────────────────────────────────────────
# MAIN SIMULATION FUNCTION
# ──────────────────────────────────────────────────────────────

def simulate_returns(df: pd.DataFrame, cat_stats: pd.DataFrame,
                     cust_hist: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """
    Main simulation loop.

    For each order sampled for return:
    - Assign fraud or genuine archetype
    - Fill in all engineered features
    - Inject label noise at NOISE_RATE
    """

    # Merge category stats and customer history
    df = df.merge(cat_stats, on="product_category_name", how="left")
    df = df.merge(cust_hist,  on="customer_unique_id",    how="left")

    # Sample orders for return
    n_returns = int(len(df) * RETURN_RATE)
    return_idx = rng.choice(df.index, size=n_returns, replace=False)
    return_df  = df.loc[return_idx].copy()

    print(f"\n  Simulating {n_returns:,} return events …")

    # Determine fraud vs. genuine split
    n_fraud   = int(n_returns * FRAUD_RATIO)
    n_genuine = n_returns - n_fraud

    fraud_idx   = return_idx[:n_fraud]
    genuine_idx = return_idx[n_fraud:]

    # Fraud archetype pools
    fraud_archetypes = [
        _wardrobing_archetype,
        _false_inr_archetype,
        _serial_returner_archetype,
        _new_account_arbitrage_archetype,
    ]
    # Weight wardrobing and false-INR slightly higher
    fraud_weights = [0.30, 0.30, 0.25, 0.15]

    # Genuine archetype pools
    genuine_archetypes = [
        _real_delay_archetype,
        _defective_item_archetype,
        _size_mismatch_archetype,
    ]
    genuine_weights = [0.35, 0.40, 0.25]

    records = []

    # ── Fraud records ──────────────────────────────────────────
    for idx in tqdm(fraud_idx, desc="  Fraud archetypes  ", ncols=80):
        row   = df.loc[idx]
        fn    = rng.choice(fraud_archetypes, p=fraud_weights)
        extra = fn(row, rng)

        cat = row.get("product_category_name", "unknown")

        record = {
            "order_id"                        : row["order_id"],
            "customer_unique_id"              : row["customer_unique_id"],
            "product_category_name"           : cat,
            "total_price"                     : _safe_float(row.get("total_price"), 0.0),
            "total_freight"                   : _safe_float(row.get("total_freight"), 0.0),
            "freight_ratio"                   : _safe_float(row.get("freight_ratio"), 0.0),
            "num_items"                       : _safe_int(row.get("num_items"), 1),
            "payment_installments"            : _safe_int(row.get("payment_installments"), 1),
            "payment_type"                    : row.get("payment_type", "credit_card"),
            "is_late_delivery_real"           : _safe_int(row.get("is_late_delivery_real"), 0),
            "delivery_delay_days"             : _safe_float(row.get("delivery_delay_days"), 0.0),
            "review_score"                    : _safe_float(row.get("review_score"), 4.0),
            "carrier_confirmed_delivery"      : 1,
            "is_high_ticket"                  : int(_safe_float(row.get("total_price"), 0.0) > 150),
            "is_fit_sensitive"                : int(cat in FIT_SENSITIVE_CATEGORIES),
            "is_fragile_category"             : int(cat in FRAGILE_CATEGORIES),
            "account_age_days"                : rng.integers(30, 1000),
            "customer_historical_return_rate" : _safe_float(row.get("customer_historical_return_rate"), 0.05),
            "product_category_return_propensity": _safe_float(row.get("product_category_return_propensity"), 0.10),
            "cat_median_price"                : _safe_float(row.get("cat_median_price"), 100.0),
            "price_vs_category_median"        : _safe_float(row.get("total_price"), 100.0) /
                                                (_safe_float(row.get("cat_median_price"), 100.0) + 1e-6),
        }
        record.update(extra)
        records.append(record)

    # ── Genuine records ────────────────────────────────────────
    for idx in tqdm(genuine_idx, desc="  Genuine archetypes", ncols=80):
        row   = df.loc[idx]
        fn    = rng.choice(genuine_archetypes, p=genuine_weights)
        extra = fn(row, rng)

        cat = row.get("product_category_name", "unknown")

        record = {
            "order_id"                        : row["order_id"],
            "customer_unique_id"              : row["customer_unique_id"],
            "product_category_name"           : cat,
            "total_price"                     : _safe_float(row.get("total_price"), 0.0),
            "total_freight"                   : _safe_float(row.get("total_freight"), 0.0),
            "freight_ratio"                   : _safe_float(row.get("freight_ratio"), 0.0),
            "num_items"                       : _safe_int(row.get("num_items"), 1),
            "payment_installments"            : _safe_int(row.get("payment_installments"), 1),
            "payment_type"                    : row.get("payment_type", "credit_card"),
            "is_late_delivery_real"           : _safe_int(row.get("is_late_delivery_real"), 0),
            "delivery_delay_days"             : _safe_float(row.get("delivery_delay_days"), 0.0),
            "review_score"                    : _safe_float(row.get("review_score"), 2.0),
            "carrier_confirmed_delivery"      : int(rng.random() > 0.05),
            "is_high_ticket"                  : int(_safe_float(row.get("total_price"), 0.0) > 150),
            "is_fit_sensitive"                : int(cat in FIT_SENSITIVE_CATEGORIES),
            "is_fragile_category"             : int(cat in FRAGILE_CATEGORIES),
            "account_age_days"                : rng.integers(90, 2000),
            "customer_historical_return_rate" : _safe_float(row.get("customer_historical_return_rate"), 0.05),
            "product_category_return_propensity": _safe_float(row.get("product_category_return_propensity"), 0.10),
            "cat_median_price"                : _safe_float(row.get("cat_median_price"), 100.0),
            "price_vs_category_median"        : _safe_float(row.get("total_price"), 100.0) /
                                                (_safe_float(row.get("cat_median_price"), 100.0) + 1e-6),
        }
        record.update(extra)
        records.append(record)

    result = pd.DataFrame(records)

    # ── Fill missing columns with defaults ─────────────────────
    defaults = {
        "days_to_return"              : 7,
        "return_window_days"          : 30,
        "complaint_category"          : "Vague/Remorse",
        "complaint_text"              : "No comment provided",
        "carrier_confirmed_delivery"  : 0,
        "is_high_ticket"              : 0,
        "is_fit_sensitive"            : 0,
        "is_fragile_category"         : 0,
        "account_age_days"            : 180,
        "archetype"                   : "Unknown",
        "label"                       : 0,
    }
    for col, default in defaults.items():
        if col not in result.columns:
            result[col] = default
        else:
            result[col] = result[col].fillna(default)

    # ── NLP numeric label ──────────────────────────────────────
    result["nlp_label"] = result["complaint_category"].map(NLP_LABEL_MAP).fillna(4).astype(int)

    # ── Price vs. category median ──────────────────────────────
    result["price_vs_category_median"] = result["price_vs_category_median"].fillna(1.0).clip(0, 20)

    # ── days_to_return_ratio ───────────────────────────────────
    result["days_to_return_ratio"] = (
        result["days_to_return"] / (result["return_window_days"] + 1e-6)
    ).clip(0, 1)

    # ──────────────────────────────────────────────────────────
    # LABEL NOISE INJECTION
    # ──────────────────────────────────────────────────────────
    print(f"\n  Injecting {NOISE_RATE*100:.0f}% label noise …")
    noise_mask = rng.random(len(result)) < NOISE_RATE
    result.loc[noise_mask, "label"] = 1 - result.loc[noise_mask, "label"]
    result["is_noisy"] = noise_mask.astype(int)

    # ──────────────────────────────────────────────────────────
    # HIGH-TICKET OVERRIDE for wardrobing
    # ──────────────────────────────────────────────────────────
    wardrobe_mask = result["archetype"] == "Wardrobing"
    result.loc[wardrobe_mask & (result["total_price"] < 100), "is_high_ticket"] = 0

    # Force carrier_confirmed_delivery = 1 for all False INR fraud
    inr_fraud = (result["archetype"] == "False INR") & (result["label"] == 1)
    result.loc[inr_fraud, "carrier_confirmed_delivery"] = 1

    return result.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Returns Fraud Simulator — Olist Dataset")
    print("=" * 60)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_STATE)
    random.seed(RANDOM_STATE)

    # 1. Load Olist data
    print("\n[1/5] Loading Olist CSV files …")
    frames = _load_olist()

    # 2. Build base transaction frame
    print("\n[2/5] Building base transaction frame …")
    base_df = _build_base_frame(frames)
    print(f"  Base frame: {len(base_df):,} delivered orders")

    # 3. Category & customer stats
    print("\n[3/5] Computing category & customer statistics …")
    cat_stats  = _compute_category_stats(base_df)
    cust_hist  = _compute_customer_history(base_df, rng)

    # 4. Simulate returns
    print("\n[4/5] Simulating returns …")
    returns_df = simulate_returns(base_df, cat_stats, cust_hist, rng)

    # Summary stats
    n_total  = len(returns_df)
    n_fraud  = returns_df["label"].sum()
    n_genuine = n_total - n_fraud
    n_noise  = returns_df["is_noisy"].sum()

    print(f"\n  ── Simulation Summary ──────────────────────────────")
    print(f"  Total return records : {n_total:>8,}")
    print(f"  Fraud (label=1)      : {n_fraud:>8,}  ({100*n_fraud/n_total:.1f}%)")
    print(f"  Genuine (label=0)    : {n_genuine:>8,}  ({100*n_genuine/n_total:.1f}%)")
    print(f"  Noisy labels         : {n_noise:>8,}  ({100*n_noise/n_total:.1f}%)")
    print(f"  ────────────────────────────────────────────────────")

    archetype_counts = returns_df.groupby("archetype")["label"].agg(["count","sum"])
    print("\n  Returns by Archetype:")
    print(archetype_counts.to_string())

    # 5. Save
    print("\n[5/5] Saving processed dataset …")
    out_path = PROCESSED_DIR / "returns_dataset.csv"
    returns_df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    # Save metadata
    meta = {
        "total_records"     : int(n_total),
        "fraud_count"       : int(n_fraud),
        "genuine_count"     : int(n_genuine),
        "fraud_ratio"       : round(float(n_fraud / n_total), 4),
        "noise_rate"        : NOISE_RATE,
        "random_state"      : RANDOM_STATE,
        "archetypes"        : returns_df["archetype"].value_counts().to_dict(),
        "complaint_categories": returns_df["complaint_category"].value_counts().to_dict(),
    }
    meta_path = PROCESSED_DIR / "returns_dataset_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved → {meta_path}")
    print("\n✓ Simulation complete.")


if __name__ == "__main__":
    main()
