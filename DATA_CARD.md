# DATA CARD — Returns Fraud vs. Genuine Grievance Sorter

**Version**: 1.0.0 | **Date**: 2026-10-18 | **Maintainer**: ML Engineering Team

---

## 1. Dataset Overview

| Attribute | Value |
|-----------|-------|
| Base Data Source | Olist Brazilian E-Commerce Public Dataset (Kaggle) |
| Base License | CC BY-NC-SA 4.0 |
| Synthetic Layer | Returns generator (`simulate_returns.py`) |
| Total Simulated Returns | ~17,500 (≈ 17.5% of ≈100k delivered orders) |
| Fraud Records | ~35% of returns |
| Genuine Records | ~65% of returns |
| Label Noise Injected | 12% |
| Random State | 42 (fully deterministic) |

---

## 2. Real Olist Data Attributes Used

These columns come directly from the Olist CSVs and are **not synthesised**:

| Column | Source File | Description |
|--------|-------------|-------------|
| `order_id` | `olist_orders_dataset.csv` | Unique order identifier |
| `customer_id` | `olist_orders_dataset.csv` | Hashed customer ID (session-level) |
| `customer_unique_id` | `olist_customers_dataset.csv` | True unique customer identifier |
| `order_purchase_timestamp` | `olist_orders_dataset.csv` | Order creation time |
| `order_delivered_customer_date` | `olist_orders_dataset.csv` | Actual delivery timestamp |
| `order_estimated_delivery_date` | `olist_orders_dataset.csv` | Expected delivery date |
| `price` | `olist_order_items_dataset.csv` | Item price (BRL) |
| `freight_value` | `olist_order_items_dataset.csv` | Shipping cost (BRL) |
| `payment_value` | `olist_order_payments_dataset.csv` | Total payment amount |
| `payment_installments` | `olist_order_payments_dataset.csv` | Number of installments |
| `payment_type` | `olist_order_payments_dataset.csv` | credit_card / boleto / voucher |
| `review_score` | `olist_order_reviews_dataset.csv` | 1–5 star rating |
| `review_comment_message` | `olist_order_reviews_dataset.csv` | Free-text complaint |
| `product_category_name` | `olist_products_dataset.csv` | Product category (Portuguese) |

### Derived from Real Data (not synthetic)

| Derived Column | Formula |
|----------------|---------|
| `is_late_delivery_real` | `delivered_date > estimated_delivery_date` |
| `delivery_delay_days` | `(delivered - estimated).days.clip(min=0)` |
| `freight_ratio` | `total_freight / (total_price + ε)` |
| `price_vs_category_median` | `order_price / category_median_price` |

---

## 3. Synthetic Return Vectors

The following columns are **fully synthesised** by `simulate_returns.py` and do **not** exist in the original Olist data:

| Synthetic Column | Generation Method | Notes |
|-----------------|-------------------|-------|
| `days_to_return` | Archetype-specific RNG (see §4) | Days from delivery to return request |
| `return_window_days` | Fixed at 30 | Standard e-commerce return policy |
| `days_to_return_ratio` | `days_to_return / return_window_days` | Normalised timing signal |
| `complaint_text` | Template sampling from 5-class pool | English synthetic text |
| `complaint_category` | Assigned per archetype | Defective / Wrong Item / Transit Damage / False INR / Vague/Remorse |
| `nlp_label` | Numeric encoding of complaint_category | 0–4 |
| `carrier_confirmed_delivery` | Set=1 for False INR fraud | Simulates carrier telemetry |
| `is_high_ticket` | `total_price > 150 BRL` | Computed threshold |
| `account_age_days` | Fraud: Uniform(0,14); Genuine: Uniform(90,2000) | Simulated account history |
| `customer_historical_return_rate` | Log-normal(μ=-3,σ=1); serial: Uniform(0.4,0.9) | Per-customer return rate |
| `product_category_return_propensity` | Rule-based by category group | Fashion=0.28, Electronics=0.18 |
| `label` | Archetype assignment + 12% noise flip | Ground-truth fraud label (0/1) |
| `is_noisy` | Boolean mask of noise-flipped labels | For research auditability |
| `archetype` | Archetype name string | For EDA and debugging |

---

## 4. Fraud Archetype Generation Rules

### 4.1 Wardrobing
- **Target category**: High-ticket luxury/apparel (`watches_gifts`, `fashion_*`, etc.)
- **`days_to_return`**: Uniform(27, 29) — near end of return window
- **`complaint_category`**: `Vague/Remorse`
- **`is_high_ticket`**: 1 (enforced)
- **Prevalence**: 30% of simulated fraud cases

### 4.2 False INR (Item Not Received)
- **`days_to_return`**: Uniform(1, 4) — very rapid
- **`complaint_category`**: `False INR`
- **`carrier_confirmed_delivery`**: 1 (contradicts claim)
- **`is_late_delivery_real`**: 0
- **Prevalence**: 30% of simulated fraud cases

### 4.3 Serial Returner
- **Customer cluster**: 3% of customers with `historical_return_rate` ∈ [0.40, 0.90]
- **`days_to_return`**: Uniform(3, 19) — distributed naturally
- **`complaint_category`**: Random across all 5 categories
- **Prevalence**: 25% of simulated fraud cases

### 4.4 New Account Arbitrage
- **`account_age_days`**: Uniform(0, 13) — very fresh account
- **`days_to_return`**: Uniform(0, 2) — instant refund request
- **`is_high_ticket`**: 1 (enforced)
- **`complaint_category`**: `Vague/Remorse`
- **Prevalence**: 15% of simulated fraud cases

---

## 5. Genuine Grievance Generation Rules

### 5.1 Real Carrier Delay
- **`is_late_delivery_real`**: 1 (inherited from real Olist data)
- **`delivery_delay_days`**: Uniform(3, 20)
- **`complaint_category`**: `Transit Damage`
- **`days_to_return`**: Uniform(2, 14)
- **Prevalence**: 35% of genuine cases

### 5.2 Defective Item
- **`review_score`**: 1 or 2 (from real Olist reviews or synthetic)
- **`complaint_category`**: `Defective`
- **`days_to_return`**: Uniform(1, 9)
- **Prevalence**: 40% of genuine cases

### 5.3 Size / Category Mismatch
- **`is_fit_sensitive`**: 1 (fashion/shoe categories)
- **`complaint_category`**: `Wrong Item`
- **`days_to_return`**: Uniform(2, 11)
- **Prevalence**: 25% of genuine cases

---

## 6. Label Noise Injection

To reflect real-world label ambiguity:
- **12% of all labels are randomly flipped** (0→1 or 1→0)
- This simulates:
  - Honest customers who return late (wardrobing-like) for genuine reasons
  - Fraudsters who mimic patient, polite buyers with low review scores
  - Human annotation uncertainty in real fraud ops datasets
- The `is_noisy` column records which rows were flipped (for research)
- **Ground truth is not perfectly observable** — the model must learn from noisy labels

---

## 7. Splitting Strategy

| Split | Method | Fraction |
|-------|--------|----------|
| Train | GroupShuffleSplit on `customer_unique_id` | 70% |
| Validation | GroupShuffleSplit on `customer_unique_id` | 15% |
| Test | GroupShuffleSplit on `customer_unique_id` | 15% |

> **Why customer-grouped splitting?** A single customer may appear in multiple orders. 
> If the same customer is in both train and test, the model can memorise customer-level 
> patterns and inflate metrics. Grouping by `customer_unique_id` ensures no data leakage 
> from repeat buyers — critical for realistic performance estimates.

---

## 8. Known Limitations & Biases

| Limitation | Impact | Mitigation |
|------------|--------|------------|
| English complaint templates (Olist is Brazilian Portuguese) | NLP model may not generalise to Portuguese | Use multilingual model or translate data in production |
| Synthetic carrier telemetry | `carrier_confirmed_delivery` is simulated, not real | Integrate actual carrier API feeds in production |
| Static category return propensity | Fashion=0.28 is a rough estimate | Update from real historical return rates |
| 12% noise rate is a global average | Real noise varies by archetype | Archetype-specific noise in v2 |
| No temporal drift simulation | Fraud patterns evolve over time | Implement concept drift monitoring |
| Olist covers 2016–2018 Brazil | Consumer behaviour has shifted | Retrain on current marketplace data |

---

## 9. Feature Importance Notes

Top predictive features (from SHAP analysis on held-out test set):

1. `customer_historical_return_rate` — strongest individual signal
2. `inr_carrier_confirmed` — highest precision for False INR archetype
3. `days_to_return_ratio` — primary wardrobing signal
4. `high_ticket_new_account` — new account arbitrage detection
5. `serial_returner_flag` — cluster-level signal
6. `price_vs_category_median` — price anomaly vs. category norm
7. `review_score` — genuine grievance corroboration signal

---

## 10. Ethical Considerations

- **No demographic data** is used as a feature (no age, gender, race, geography beyond national Olist scope)
- **Asymmetric cost weighting** deliberately biases toward protecting honest customers (5× FP penalty)
- **SHAP explainability** ensures every decision can be audited and explained to the customer
- **Amber channel** (Quick check) minimises outright blocking — prefers soft friction (photo upload, OTP pickup)
- **Label noise** is documented and preserved in `is_noisy` column for research reproducibility
