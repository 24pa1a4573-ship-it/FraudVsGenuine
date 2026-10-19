"""
app.py
────────────────────────────────────────────────────────────────
Interactive Streamlit dashboard for the Returns Fraud vs.
Genuine Grievance Sorter.

Features:
  • Preset selector (4 archetypes)
  • Custom return submission form
  • Risk gauge meter + triage badge
  • SHAP waterfall / bar chart
  • Real-time cost-ratio slider (dynamic threshold shift)
  • Full inference log

Run:
    streamlit run app.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

# ──────────────────────────────────────────────────────────────
# PATH SETUP  (allow `src.*` imports when run from project root)
# ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import ReturnSorterPipeline, PRESET_SCENARIOS, TriageResult

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Returns Fraud Sorter",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────
# GLOBAL CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Font ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── App background ─────────────────────────────────────── */
.stApp {
    background: linear-gradient(135deg, #0a0e1a 0%, #0d1526 50%, #0a1020 100%);
    color: #e2e8f0;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1526 0%, #0a0e1a 100%);
    border-right: 1px solid rgba(99,179,237,0.15);
}

/* ── Main header ─────────────────────────────────────────── */
.main-header {
    background: linear-gradient(135deg, #1a2744 0%, #0f1e36 100%);
    border: 1px solid rgba(99,179,237,0.25);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}

.main-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #63b3ed, #9f7aea, #ed64a6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
}

.main-header p {
    color: #94a3b8;
    font-size: 1.05rem;
    margin-top: 0.5rem;
}

/* ── Cards ───────────────────────────────────────────────── */
.card {
    background: linear-gradient(135deg, #1a2744 0%, #141f36 100%);
    border: 1px solid rgba(99,179,237,0.18);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(0,0,0,0.4);
}

/* ── Badge styles ────────────────────────────────────────── */
.badge-fraud {
    background: linear-gradient(135deg, #c53030, #9b2c2c);
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
    border: 1px solid rgba(252,129,74,0.4);
    box-shadow: 0 0 20px rgba(197,48,48,0.4);
    animation: pulse-red 2s infinite;
}

.badge-genuine {
    background: linear-gradient(135deg, #276749, #22543d);
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
    border: 1px solid rgba(72,187,120,0.4);
    box-shadow: 0 0 20px rgba(39,103,73,0.4);
}

.badge-unclear {
    background: linear-gradient(135deg, #c05621, #9c4221);
    color: white;
    border-radius: 8px;
    padding: 0.5rem 1.2rem;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
    border: 1px solid rgba(246,173,85,0.4);
    box-shadow: 0 0 20px rgba(192,86,33,0.4);
}

@keyframes pulse-red {
    0%, 100% { box-shadow: 0 0 20px rgba(197,48,48,0.4); }
    50%       { box-shadow: 0 0 35px rgba(197,48,48,0.7); }
}

/* ── Metric tiles ────────────────────────────────────────── */
.metric-tile {
    background: linear-gradient(135deg, #1e3048 0%, #172438 100%);
    border: 1px solid rgba(99,179,237,0.2);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-tile .metric-value {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #63b3ed, #9f7aea);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.metric-tile .metric-label {
    color: #94a3b8;
    font-size: 0.85rem;
    margin-top: 0.3rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* ── Reason list ─────────────────────────────────────────── */
.reason-item {
    background: rgba(26, 39, 68, 0.6);
    border-left: 3px solid #63b3ed;
    border-radius: 0 8px 8px 0;
    padding: 0.6rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.92rem;
    color: #cbd5e0;
}

.reason-item.risk-high { border-left-color: #fc8181; }
.reason-item.risk-low  { border-left-color: #68d391; }

/* ── Action banner ───────────────────────────────────────── */
.action-banner-approve {
    background: linear-gradient(135deg, #22543d, #276749);
    border: 1px solid rgba(72,187,120,0.4);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    font-size: 1.2rem;
    font-weight: 600;
    color: #9ae6b4;
}

.action-banner-check {
    background: linear-gradient(135deg, #7b341e, #9c4221);
    border: 1px solid rgba(246,173,85,0.4);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    font-size: 1.2rem;
    font-weight: 600;
    color: #fbd38d;
}

.action-banner-manual {
    background: linear-gradient(135deg, #742a2a, #9b2c2c);
    border: 1px solid rgba(252,129,74,0.4);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    font-size: 1.2rem;
    font-weight: 600;
    color: #feb2b2;
    animation: pulse-red 2s infinite;
}

/* ── Sidebar section title ───────────────────────────────── */
.sidebar-section {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #63b3ed;
    margin: 1.2rem 0 0.5rem 0;
}

/* ── Score bar ───────────────────────────────────────────── */
.score-bar-container {
    background: rgba(255,255,255,0.08);
    border-radius: 999px;
    height: 12px;
    margin: 0.5rem 0;
    overflow: hidden;
}

/* ── Plotly chart background fix ─────────────────────────── */
.js-plotly-plot .plotly .bg {
    fill: transparent !important;
}

/* ── Streamlit element overrides ─────────────────────────── */
div[data-testid="stSelectbox"] label,
div[data-testid="stSlider"] label,
div[data-testid="stTextArea"] label,
div[data-testid="stNumberInput"] label,
div[data-testid="stTextInput"] label {
    color: #94a3b8 !important;
    font-size: 0.87rem !important;
    font-weight: 500 !important;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #3182ce, #2c5282) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
}

div[data-testid="stButton"] > button:hover {
    background: linear-gradient(135deg, #4299e1, #3182ce) !important;
    box-shadow: 0 4px 15px rgba(49,130,206,0.5) !important;
    transform: translateY(-1px) !important;
}

hr { border-color: rgba(99,179,237,0.15) !important; }

/* ── Hide streamlit branding ─────────────────────────────── */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# CACHED PIPELINE LOADER
# ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading fraud detection models …")
def load_pipeline() -> ReturnSorterPipeline:
    """Load and cache the inference pipeline."""
    pipeline = ReturnSorterPipeline()
    pipeline._load_models()
    return pipeline


# ──────────────────────────────────────────────────────────────
# CHART BUILDERS
# ──────────────────────────────────────────────────────────────

def build_gauge(score: float, t_low: float, t_high: float) -> go.Figure:
    """Build a Plotly gauge chart for the risk score."""
    t_low_pct  = t_low  * 100
    t_high_pct = t_high * 100

    if score < t_low_pct:
        color = "#48BB78"
        needle_color = "#9AE6B4"
    elif score < t_high_pct:
        color = "#ED8936"
        needle_color = "#FBD38D"
    else:
        color = "#FC8181"
        needle_color = "#FEB2B2"

    fig = go.Figure(go.Indicator(
        mode  = "gauge+number+delta",
        value = score,
        delta = {"reference": 50, "valueformat": ".1f"},
        number= {"font": {"size": 48, "color": color, "family": "Inter"}},
        gauge = {
            "axis"      : {
                "range"     : [0, 100],
                "tickwidth" : 1,
                "tickcolor" : "#4a5568",
                "tickfont"  : {"color": "#718096", "size": 11},
            },
            "bar"       : {"color": color, "thickness": 0.28},
            "bgcolor"   : "rgba(0,0,0,0)",
            "borderwidth": 0,
            "steps"     : [
                {"range": [0, t_low_pct],    "color": "rgba(72,187,120,0.12)"},
                {"range": [t_low_pct, t_high_pct], "color": "rgba(237,137,54,0.12)"},
                {"range": [t_high_pct, 100], "color": "rgba(252,129,74,0.12)"},
            ],
            "threshold" : {
                "line"      : {"color": needle_color, "width": 3},
                "thickness" : 0.85,
                "value"     : score,
            },
        },
        title = {
            "text"      : "Fraud Risk Score",
            "font"      : {"size": 16, "color": "#94a3b8", "family": "Inter"},
        },
    ))

    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(0,0,0,0)",
        font_color    = "#e2e8f0",
        height        = 300,
        margin        = dict(t=60, b=10, l=10, r=10),
    )
    return fig


def build_shap_bar(shap_values: list[float], feature_names: list[str]) -> go.Figure:
    """Build a horizontal bar chart of SHAP values for top features."""
    if not shap_values or not feature_names:
        return go.Figure()

    sv_arr = np.array(shap_values)
    fn_arr = np.array(feature_names)

    # Top 10 by absolute value
    top_n  = min(10, len(sv_arr))
    top_idx = np.argsort(np.abs(sv_arr))[-top_n:][::-1]

    top_sv  = sv_arr[top_idx][::-1]
    top_fn  = fn_arr[top_idx][::-1]

    # Human-readable labels
    from src.pipeline import FEATURE_DISPLAY_NAMES
    labels = [FEATURE_DISPLAY_NAMES.get(f, f).replace("_", " ").title() for f in top_fn]

    colors = [
        "rgba(252,129,74,0.85)" if v > 0 else "rgba(72,187,120,0.85)"
        for v in top_sv
    ]

    fig = go.Figure(go.Bar(
        x           = top_sv,
        y           = labels,
        orientation = "h",
        marker_color= colors,
        marker_line = dict(width=0),
        hovertemplate = "<b>%{y}</b><br>SHAP: %{x:.4f}<extra></extra>",
        text        = [f"{v:+.4f}" for v in top_sv],
        textposition= "outside",
        textfont    = {"size": 11, "color": "#94a3b8"},
    ))

    fig.add_vline(x=0, line_color="rgba(99,179,237,0.4)", line_width=1.5)

    fig.update_layout(
        title       = dict(
            text    = "🔍 SHAP Risk Driver Analysis",
            font    = {"size": 16, "color": "#cbd5e0", "family": "Inter"},
            x       = 0,
        ),
        xaxis       = dict(
            title   = "SHAP value (← reduces risk | increases risk →)",
            title_font = {"size": 12, "color": "#718096"},
            tickfont   = {"color": "#718096"},
            gridcolor  = "rgba(255,255,255,0.06)",
            zeroline   = False,
        ),
        yaxis       = dict(
            tickfont   = {"color": "#cbd5e0", "size": 11},
            gridcolor  = "rgba(255,255,255,0.04)",
        ),
        paper_bgcolor = "rgba(0,0,0,0)",
        plot_bgcolor  = "rgba(13,21,38,0.8)",
        font_color    = "#e2e8f0",
        height        = 360,
        margin        = dict(t=50, b=40, l=10, r=60),
    )
    return fig


def build_threshold_diagram(
    t_low: float, t_high: float, score: float
) -> go.Figure:
    """Horizontal threshold bar showing where the score falls."""
    fig = go.Figure()

    # Zone fills
    fig.add_shape(type="rect", x0=0, x1=t_low*100,      y0=0, y1=1,
                  fillcolor="rgba(72,187,120,0.20)",  line_width=0)
    fig.add_shape(type="rect", x0=t_low*100, x1=t_high*100, y0=0, y1=1,
                  fillcolor="rgba(237,137,54,0.20)",  line_width=0)
    fig.add_shape(type="rect", x0=t_high*100, x1=100,   y0=0, y1=1,
                  fillcolor="rgba(252,129,74,0.20)",  line_width=0)

    # Zone labels
    for x, label, color in [
        (t_low*100/2,              "✅ Auto-approve",  "#48BB78"),
        ((t_low+t_high)*50,        "⚠️ Quick check",   "#ED8936"),
        ((t_high*100+100)/2,       "🚨 Manual review", "#FC8181"),
    ]:
        fig.add_annotation(x=x, y=0.5, text=label,
                           font={"color": color, "size": 12, "family": "Inter"},
                           showarrow=False)

    # Threshold lines
    for thresh, label in [(t_low*100, f"T_low={t_low:.2f}"),
                          (t_high*100, f"T_high={t_high:.2f}")]:
        fig.add_shape(type="line", x0=thresh, x1=thresh, y0=0, y1=1,
                      line=dict(color="rgba(99,179,237,0.7)", width=2, dash="dash"))
        fig.add_annotation(x=thresh, y=1.08, text=label,
                           font={"color": "#63b3ed", "size": 10},
                           showarrow=False)

    # Score needle
    sc_color = "#FC8181" if score >= t_high*100 else ("#ED8936" if score >= t_low*100 else "#48BB78")
    fig.add_shape(type="line", x0=score, x1=score, y0=-0.1, y1=1.1,
                  line=dict(color=sc_color, width=3))
    fig.add_annotation(x=score, y=-0.25, text=f"▲ {score:.1f}",
                       font={"color": sc_color, "size": 13, "family": "Inter"},
                       showarrow=False)

    fig.update_layout(
        xaxis=dict(range=[0, 100], tickfont={"color":"#718096"}, title="Risk Score",
                   title_font={"color":"#718096","size":11}),
        yaxis=dict(range=[-0.4, 1.3], showticklabels=False, showgrid=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(13,21,38,0.8)",
        height=160,
        margin=dict(t=30, b=30, l=10, r=10),
        showlegend=False,
    )
    return fig


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────

def render_sidebar() -> tuple[dict, int, int, bool]:
    """Render sidebar controls. Returns (order_dict, fp_cost, fn_cost, run_flag)."""
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center;padding:1rem 0;">
            <div style="font-size:2.5rem">🔍</div>
            <div style="font-size:1.2rem;font-weight:700;color:#63b3ed;">Fraud Sorter</div>
            <div style="font-size:0.78rem;color:#4a5568;">Returns Triage System</div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # ── Input mode ────────────────────────────────────────
        st.markdown('<p class="sidebar-section">📋 Input Mode</p>', unsafe_allow_html=True)
        input_mode = st.selectbox(
            "Source",
            ["🎭 Preset Scenario", "✏️ Custom Return"],
            label_visibility="collapsed",
        )

        order_dict = {}

        if "Preset" in input_mode:
            preset_names = list(PRESET_SCENARIOS.keys())
            preset_labels = [PRESET_SCENARIOS[k]["label"] for k in preset_names]
            selected_label = st.selectbox("Choose Scenario", preset_labels)
            selected_key   = preset_names[preset_labels.index(selected_label)]
            order_dict     = dict(PRESET_SCENARIOS[selected_key])

            st.markdown(f"""
            <div class="card" style="margin-top:0.5rem;padding:0.8rem 1rem;">
                <div style="font-size:0.78rem;color:#94a3b8;">Scenario</div>
                <div style="font-size:0.88rem;color:#cbd5e0;margin-top:0.3rem;">
                    {PRESET_SCENARIOS[selected_key]['description']}
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            # Custom form
            st.markdown('<p class="sidebar-section">📦 Order Details</p>', unsafe_allow_html=True)

            order_dict["total_price"]        = st.number_input("Order Value (BRL)", 0.0, 10000.0, 250.0, 10.0)
            order_dict["total_freight"]      = st.number_input("Freight (BRL)",     0.0, 500.0,   20.0,   1.0)
            order_dict["cat_median_price"]   = st.number_input("Category Median Price (BRL)", 10.0, 5000.0, 200.0, 10.0)
            order_dict["num_items"]          = st.number_input("Number of Items",  1, 20, 1)
            order_dict["payment_installments"]= st.slider("Payment Installments", 1, 24, 1)

            st.markdown('<p class="sidebar-section">📅 Return Details</p>', unsafe_allow_html=True)
            order_dict["days_to_return"]     = st.slider("Days to Return Filed",  0, 60, 7)
            order_dict["return_window_days"] = st.slider("Return Window (Days)",  7, 90, 30)

            st.markdown('<p class="sidebar-section">🚚 Delivery</p>', unsafe_allow_html=True)
            order_dict["is_late_delivery_real"]= st.toggle("Real Carrier Delay?", False)
            order_dict["delivery_delay_days"]  = st.number_input("Delay Days", 0.0, 60.0, 0.0, 1.0)
            order_dict["carrier_confirmed_delivery"] = st.toggle("Carrier Confirmed Delivery?", True)

            st.markdown('<p class="sidebar-section">👤 Customer</p>', unsafe_allow_html=True)
            order_dict["account_age_days"]          = st.number_input("Account Age (Days)", 0, 3000, 365, 10)
            order_dict["customer_historical_return_rate"] = st.slider("Historical Return Rate", 0.0, 1.0, 0.05, 0.01)
            order_dict["review_score"]              = st.slider("Review Score (1-5)", 1.0, 5.0, 3.0, 0.5)

            st.markdown('<p class="sidebar-section">🏷️ Product</p>', unsafe_allow_html=True)
            order_dict["is_high_ticket"]   = st.toggle("High-Ticket Item?",      False)
            order_dict["is_fit_sensitive"] = st.toggle("Fit-Sensitive Category?", False)
            order_dict["is_fragile_category"]= st.toggle("Fragile Category?",    False)
            order_dict["product_category_return_propensity"] = st.slider("Category Return Propensity", 0.0, 1.0, 0.12, 0.01)

            st.markdown('<p class="sidebar-section">💬 Complaint</p>', unsafe_allow_html=True)
            cat_options = ["Vague/Remorse", "False INR", "Defective", "Wrong Item", "Transit Damage"]
            order_dict["complaint_category"] = st.selectbox("Category", cat_options)
            order_dict["complaint_text"]     = st.text_area(
                "Complaint Text",
                "No comment provided.",
                height=80,
            )

        st.divider()

        # ── Cost ratio slider ─────────────────────────────────
        st.markdown('<p class="sidebar-section">⚖️ Cost-Ratio Tuning</p>', unsafe_allow_html=True)
        st.markdown(
            "<div style='font-size:0.78rem;color:#718096;margin-bottom:0.5rem;'>"
            "How many times costlier is blocking a genuine customer vs. missing fraud?"
            "</div>",
            unsafe_allow_html=True,
        )
        fp_cost = st.slider("FP Penalty Multiplier", 1, 20, 5, 1,
                             help="Default: 5 (CLV destruction). Raise to be more lenient.")
        fn_cost = 1

        st.markdown(f"""
        <div style="background:rgba(26,39,68,0.6);border-radius:8px;padding:0.6rem 0.9rem;
                    font-size:0.8rem;color:#94a3b8;margin-top:0.3rem;">
            Cost function: <b style="color:#63b3ed">({fp_cost}×FP) + (1×FN)</b>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        run_flag = st.button("🔍 Analyze Return", use_container_width=True)

    return order_dict, fp_cost, fn_cost, run_flag


# ──────────────────────────────────────────────────────────────
# RESULT RENDERING
# ──────────────────────────────────────────────────────────────

def render_result(result: TriageResult):
    """Render the full triage result in the main panel."""

    score  = result.risk_score
    cat    = result.category
    action = result.recommended_action

    # ── Category badge & action banner ────────────────────────
    badge_map = {
        "Likely Fraud"       : ("badge-fraud",   "🚨 Likely Fraud"),
        "Genuine Grievance"  : ("badge-genuine", "✅ Genuine Grievance"),
        "Unclear"            : ("badge-unclear", "⚠️ Unclear"),
    }
    action_map = {
        "Auto-approve"  : ("action-banner-approve", "✅ Recommended Action: Auto-Approve"),
        "Quick check"   : ("action-banner-check",   "⚠️ Recommended Action: Quick Check (Photo/OTP)"),
        "Manual review" : ("action-banner-manual",  "🚨 Recommended Action: Manual Review by Fraud Ops"),
    }

    badge_cls, badge_txt  = badge_map.get(cat,    ("badge-unclear", cat))
    action_cls, action_txt = action_map.get(action, ("action-banner-check", action))

    col_score, col_cat, col_action = st.columns([1, 1, 1])

    with col_score:
        st.markdown(f"""
        <div class="metric-tile">
            <div class="metric-value">{score:.1f}</div>
            <div class="metric-label">Risk Score / 100</div>
        </div>
        """, unsafe_allow_html=True)

    with col_cat:
        st.markdown(f"""
        <div style="height:100%;display:flex;align-items:center;justify-content:center;padding:1rem;">
            <span class="{badge_cls}">{badge_txt}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_action:
        st.markdown(f"""
        <div class="{action_cls}">{action_txt}</div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Gauge + Threshold diagram ──────────────────────────────
    col_gauge, col_thresh = st.columns([1, 1.4])

    with col_gauge:
        st.plotly_chart(
            build_gauge(score, result.t_low, result.t_high),
            use_container_width=True,
        )

    with col_thresh:
        st.markdown("""
        <div style="font-size:0.85rem;color:#94a3b8;margin-bottom:0.4rem;font-weight:500;">
            📊 Decision Boundary
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(
            build_threshold_diagram(result.t_low, result.t_high, score),
            use_container_width=True,
        )

        # Score breakdown
        st.markdown(f"""
        <div style="display:flex;gap:0.8rem;margin-top:0.3rem;">
            <div class="metric-tile" style="flex:1;padding:0.8rem;">
                <div style="font-size:1.4rem;font-weight:700;color:#63b3ed;">
                    {result.behavior_prob*100:.1f}
                </div>
                <div class="metric-label">Behavioral Model %</div>
            </div>
            <div class="metric-tile" style="flex:1;padding:0.8rem;">
                <div style="font-size:1.4rem;font-weight:700;color:#9f7aea;">
                    {result.nlp_risk_score*100:.1f}
                </div>
                <div class="metric-label">NLP Risk %</div>
            </div>
            <div class="metric-tile" style="flex:1;padding:0.8rem;">
                <div style="font-size:1rem;font-weight:700;color:#ed64a6;">
                    {result.nlp_complaint_category}
                </div>
                <div class="metric-label">Complaint Type</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── SHAP bar chart + top reasons ──────────────────────────
    col_shap, col_reasons = st.columns([1.3, 1])

    with col_shap:
        if result.shap_values:
            st.plotly_chart(
                build_shap_bar(result.shap_values, result.shap_feature_names),
                use_container_width=True,
            )
        else:
            st.info("SHAP values unavailable for this prediction.")

    with col_reasons:
        st.markdown("""
        <div style="font-size:0.9rem;font-weight:600;color:#cbd5e0;margin-bottom:0.8rem;">
            🧠 Top Risk Drivers (Plain English)
        </div>
        """, unsafe_allow_html=True)

        for i, reason in enumerate(result.top_reasons):
            is_risk = any(kw in reason for kw in ["↑", "increases", "High", "Serial", "New account",
                                                   "Confirmed", "Late return", "INR claim"])
            css_cls = "risk-high" if is_risk else "risk-low"
            icon    = "🔴" if is_risk else "🟢"
            st.markdown(f"""
            <div class="reason-item {css_cls}">
                {icon} {reason}
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Full inference details expander ───────────────────────
    with st.expander("📋 Full Inference Details", expanded=False):
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.markdown(f"""
            | Parameter | Value |
            |-----------|-------|
            | Risk Score | **{score:.2f}** |
            | Category | **{cat}** |
            | Action | **{action}** |
            | Behavior Probability | {result.behavior_prob:.4f} |
            | NLP Risk Score | {result.nlp_risk_score:.4f} |
            | NLP Complaint Category | {result.nlp_complaint_category} |
            """)
        with col_d2:
            st.markdown(f"""
            | Threshold | Value |
            |-----------|-------|
            | T_low (approve cutoff) | {result.t_low:.4f} |
            | T_high (fraud cutoff) | {result.t_high:.4f} |
            | Composite Formula | 0.65×Behavioral + 0.35×NLP |
            """)

        if result.shap_values and result.shap_feature_names:
            st.markdown("**All SHAP Values**")
            shap_df = pd.DataFrame({
                "Feature" : result.shap_feature_names,
                "SHAP Value": result.shap_values,
            }).sort_values("SHAP Value", key=abs, ascending=False)
            st.dataframe(shap_df, height=250, use_container_width=True)


# ──────────────────────────────────────────────────────────────
# ABOUT / HEADER
# ──────────────────────────────────────────────────────────────

def render_header():
    st.markdown("""
    <div class="main-header">
        <h1>🔍 Returns Fraud vs. Genuine Grievance Sorter</h1>
        <p>
            AI-powered triage system · LightGBM + NLP · SHAP explainability ·
            Cost-optimized thresholding (5× FP:FN penalty)
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_model_status():
    """Show model status or training instructions."""
    models_dir = Path("models")
    required   = ["lgbm_model.pkl", "nlp_pipeline.pkl", "shap_explainer.pkl",
                   "feature_columns.json", "thresholds.json"]
    missing    = [f for f in required if not (models_dir / f).exists()]

    if missing:
        st.error(f"""
        **Models not found!** Missing: `{', '.join(missing)}`

        Run the full pipeline first:
        ```bash
        # 1. Download Olist data → data/raw/
        python src/simulate_returns.py

        # 2. Train models
        python src/train.py
        ```
        """)
        return False
    return True


def _load_real_metrics() -> dict:
    """Parse real metrics from training_report.txt if available."""
    report_path = Path("models/training_report.txt")
    metrics = {
        "pr_auc": "0.8023", "roc_auc": "0.8546",
        "heuristic_pr": "0.6334", "heuristic_roc": "0.7252",
        "test_size": "2,531", "fraud_rate": "39.08%",
        "fp": "0", "fn": "989", "business_cost": "989",
    }
    if not report_path.exists():
        return metrics
    try:
        txt = report_path.read_text(encoding="utf-8", errors="replace")
        import re
        def _get(pattern, default):
            m = re.search(pattern, txt)
            return m.group(1).strip() if m else default
        metrics["pr_auc"]        = _get(r"PR-AUC\s*:\s*([\d.]+)", metrics["pr_auc"])
        metrics["roc_auc"]       = _get(r"ROC-AUC\s*:\s*([\d.]+)", metrics["roc_auc"])
        metrics["heuristic_pr"]  = _get(r"Heuristic PR-AUC\s*:\s*([\d.]+)", metrics["heuristic_pr"])
        metrics["heuristic_roc"] = _get(r"Heuristic ROC-AUC\s*:\s*([\d.]+)", metrics["heuristic_roc"])
        metrics["test_size"]     = _get(r"Test set size\s*:\s*([\d,]+)", metrics["test_size"])
        metrics["fraud_rate"]    = _get(r"Fraud rate\s*:\s*([\d.]+%)", metrics["fraud_rate"])
        metrics["fp"]            = _get(r"False Positives\s*:\s*(\d+)", metrics["fp"])
        metrics["fn"]            = _get(r"False Negatives\s*:\s*(\d+)", metrics["fn"])
        metrics["business_cost"] = _get(r"Business Cost\s*:\s*([\d,]+)", metrics["business_cost"])
    except Exception:
        pass
    return metrics


def render_about_tab():
    """Rich About & Benchmarks tab with real metrics."""
    m = _load_real_metrics()

    # ── Hero Banner ───────────────────────────────────────────
    st.markdown("""
    <div style="background:linear-gradient(135deg,#1a2744,#0f1e36);border:1px solid rgba(99,179,237,0.25);
                border-radius:16px;padding:2rem 2.5rem;margin-bottom:2rem;
                box-shadow:0 8px 32px rgba(0,0,0,0.4);">
        <h2 style="background:linear-gradient(90deg,#63b3ed,#9f7aea,#ed64a6);
                   -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                   margin:0 0 0.5rem 0;font-size:1.8rem;">📊 About & Benchmarks</h2>
        <p style="color:#94a3b8;margin:0;font-size:1rem;">
            AI-powered triage system combining behavioural ML with NLP complaint analysis —
            trained on the <b style="color:#63b3ed">Olist e-commerce dataset</b> with 16,882 synthetic return records.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Live Metric Cards ─────────────────────────────────────
    st.markdown('<p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#63b3ed;margin-bottom:0.8rem;">🏆 Live Model Performance</p>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    for col, label, val, color in [
        (c1, "Composite PR-AUC",  m["pr_auc"],        "#63b3ed"),
        (c2, "Composite ROC-AUC", m["roc_auc"],       "#9f7aea"),
        (c3, "Heuristic PR-AUC",  m["heuristic_pr"],  "#ed64a6"),
        (c4, "Test Set Size",     m["test_size"],      "#68d391"),
        (c5, "Fraud Rate",        m["fraud_rate"],     "#fc8181"),
    ]:
        col.markdown(f"""
        <div class="metric-tile">
            <div style="font-size:2rem;font-weight:700;color:{color};">{val}</div>
            <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Benchmark Table (with plotly bar) ─────────────────────
    col_bench, col_chart = st.columns([1, 1.2])

    with col_bench:
        st.markdown('<p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#63b3ed;margin-bottom:0.8rem;">📈 Model Comparison</p>', unsafe_allow_html=True)
        try:
            comp_pr  = float(m["pr_auc"])
            heir_pr  = float(m["heuristic_pr"])
            comp_roc = float(m["roc_auc"])
            heir_roc = float(m["heuristic_roc"])
        except ValueError:
            comp_pr, heir_pr, comp_roc, heir_roc = 0.80, 0.63, 0.85, 0.73

        bench_fig = go.Figure()
        models    = ["Heuristic Rules", "Composite ML"]
        pr_vals   = [heir_pr, comp_pr]
        roc_vals  = [heir_roc, comp_roc]
        bench_fig.add_trace(go.Bar(name="PR-AUC",  x=models, y=pr_vals,
                                   marker_color=["rgba(99,179,237,0.5)","rgba(99,179,237,0.9)"],
                                   text=[f"{v:.4f}" for v in pr_vals], textposition="outside"))
        bench_fig.add_trace(go.Bar(name="ROC-AUC", x=models, y=roc_vals,
                                   marker_color=["rgba(159,122,234,0.5)","rgba(159,122,234,0.9)"],
                                   text=[f"{v:.4f}" for v in roc_vals], textposition="outside"))
        bench_fig.update_layout(
            barmode="group", height=300,
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,21,38,0.8)",
            font_color="#e2e8f0", legend=dict(font=dict(color="#94a3b8")),
            yaxis=dict(range=[0, 1.05], tickfont=dict(color="#718096"), gridcolor="rgba(255,255,255,0.06)"),
            xaxis=dict(tickfont=dict(color="#cbd5e0")),
            margin=dict(t=20, b=20, l=10, r=10),
        )
        st.plotly_chart(bench_fig, use_container_width=True)

    with col_chart:
        st.markdown('<p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#63b3ed;margin-bottom:0.8rem;">🔍 Confusion Matrix Breakdown</p>', unsafe_allow_html=True)
        try:
            fp_v = int(m["fp"].replace(",", ""))
            fn_v = int(m["fn"].replace(",", ""))
        except (ValueError, AttributeError):
            fp_v, fn_v = 0, 989
        cm_fig = go.Figure(go.Heatmap(
            z=[[1542-fp_v, fp_v], [fn_v, 989-fn_v]],
            x=["Predicted Genuine", "Predicted Fraud"],
            y=["Actual Genuine", "Actual Fraud"],
            colorscale=[[0,"rgba(13,21,38,0.9)"],[0.5,"rgba(99,179,237,0.4)"],[1,"rgba(99,179,237,0.9)"]],
            text=[[f"TN: {1542-fp_v:,}",f"FP: {fp_v:,}"],[f"FN: {fn_v:,}",f"TP: {989-fn_v:,}"]],
            texttemplate="%{text}", textfont={"size":14,"color":"white"},
            showscale=False,
        ))
        cm_fig.update_layout(
            height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(13,21,38,0.8)",
            font_color="#e2e8f0", margin=dict(t=10, b=10, l=10, r=10),
            xaxis=dict(tickfont=dict(color="#94a3b8")),
            yaxis=dict(tickfont=dict(color="#94a3b8")),
        )
        st.plotly_chart(cm_fig, use_container_width=True)

    st.divider()

    # ── System Architecture ────────────────────────────────────
    st.markdown('<p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#63b3ed;margin-bottom:1rem;">⚙️ System Architecture</p>', unsafe_allow_html=True)

    arch_cols = st.columns(3)
    arch_cards = [
        ("🧮", "Feature Engineering", "#63b3ed",
         "25 behavioural features including days-to-return ratio, serial returner score, "
         "high-ticket×new-account interaction, carrier confirmation, and NLP complaint embedding."),
        ("🌲", "LightGBM Behavioral", "#9f7aea",
         "Gradient-boosted tree model (weight=0.65). Trained with cost-sensitive class weighting. "
         "SHAP TreeExplainer provides per-prediction feature attribution for full explainability."),
        ("💬", "TF-IDF NLP Classifier", "#ed64a6",
         "Logistic regression on TF-IDF complaint text (weight=0.35). Classifies into 5 categories: "
         "Defective, Wrong Item, Transit Damage, False INR, Vague/Remorse."),
    ]
    for col, (icon, title, color, desc) in zip(arch_cols, arch_cards):
        col.markdown(f"""
        <div class="card" style="border-top:3px solid {color};">
            <div style="font-size:2rem;margin-bottom:0.5rem;">{icon}</div>
            <div style="font-size:1rem;font-weight:700;color:{color};margin-bottom:0.5rem;">{title}</div>
            <div style="font-size:0.85rem;color:#94a3b8;line-height:1.6;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Decision Flow ─────────────────────────────────────────
    flow_col = st.columns([1, 3, 1])[1]
    with flow_col:
        st.markdown("""
        <div class="card" style="text-align:center;">
            <div style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#63b3ed;margin-bottom:1rem;">Decision Pipeline</div>
            <div style="display:flex;align-items:center;justify-content:center;gap:0.5rem;flex-wrap:wrap;">
                <div style="background:rgba(99,179,237,0.15);border:1px solid rgba(99,179,237,0.4);border-radius:8px;padding:0.5rem 1rem;font-size:0.85rem;color:#63b3ed;">Return Request</div>
                <div style="color:#4a5568;font-size:1.2rem;">→</div>
                <div style="background:rgba(159,122,234,0.15);border:1px solid rgba(159,122,234,0.4);border-radius:8px;padding:0.5rem 1rem;font-size:0.85rem;color:#9f7aea;">Feature Engineering</div>
                <div style="color:#4a5568;font-size:1.2rem;">→</div>
                <div style="background:rgba(99,179,237,0.15);border:1px solid rgba(99,179,237,0.4);border-radius:8px;padding:0.5rem 1rem;font-size:0.85rem;color:#63b3ed;">LightGBM (65%)</div>
                <div style="color:#4a5568;font-size:0.9rem;">+</div>
                <div style="background:rgba(237,100,166,0.15);border:1px solid rgba(237,100,166,0.4);border-radius:8px;padding:0.5rem 1rem;font-size:0.85rem;color:#ed64a6;">NLP (35%)</div>
                <div style="color:#4a5568;font-size:1.2rem;">→</div>
                <div style="background:rgba(104,211,145,0.15);border:1px solid rgba(104,211,145,0.4);border-radius:8px;padding:0.5rem 1rem;font-size:0.85rem;color:#68d391;">Triage Decision</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Fraud Archetypes ──────────────────────────────────────
    st.markdown('<p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#63b3ed;margin-bottom:1rem;">🗂️ Fraud Archetypes Detected</p>', unsafe_allow_html=True)

    archetypes = [
        ("🛍️", "Wardrobing",              "#fc8181", "🔴 High",
         "Returns luxury/fashion items after use. Signal: late return window + vague reason + high-ticket."),
        ("📦", "False INR",              "#fc8181", "🔴 High",
         "Claims item never arrived despite carrier confirmation. Signal: carrier_confirmed=True + non-receipt text."),
        ("🔁", "Serial Returner",        "#f6ad55", "🟠 Med-High",
         "Exploits return policy repeatedly. Signal: high historical return rate across orders."),
        ("🆕", "New Account Arbitrage",  "#fc8181", "🔴 High",
         "New account + high-value item + instant refund request. Signal: account_age < 30 days."),
        ("🚚", "Real Carrier Delay",     "#68d391", "🟢 Low",
         "Genuine complaint caused by late delivery. Signal: is_late_delivery_real + damage text."),
        ("🔧", "Defective Item",         "#68d391", "🟢 Low",
         "Authentic product defect. Signal: 1-2 star review + defect/broken keywords in complaint."),
        ("👗", "Size/Category Mismatch", "#68d391", "🟢 Low",
         "Honest size or fit issue. Signal: fit-sensitive category + size/fit complaint keywords."),
    ]

    arch_row1 = st.columns(4)
    arch_row2 = st.columns(3)
    all_cols  = arch_row1 + arch_row2
    for col, (icon, name, color, risk, desc) in zip(all_cols, archetypes):
        col.markdown(f"""
        <div class="card" style="border-left:3px solid {color};min-height:160px;">
            <div style="font-size:1.5rem;margin-bottom:0.4rem;">{icon}</div>
            <div style="font-weight:700;color:{color};font-size:0.9rem;margin-bottom:0.2rem;">{name}</div>
            <div style="font-size:0.75rem;color:#94a3b8;margin-bottom:0.5rem;">{risk}</div>
            <div style="font-size:0.78rem;color:#718096;line-height:1.5;">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Cost Asymmetry ─────────────────────────────────────────
    st.markdown('<p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#63b3ed;margin-bottom:1rem;">⚖️ Cost Asymmetry Principle</p>', unsafe_allow_html=True)
    cost_c1, cost_c2 = st.columns(2)
    cost_c1.markdown("""
    <div class="card" style="border-top:3px solid #fc8181;">
        <div style="font-size:1.5rem;">🚫</div>
        <div style="font-weight:700;color:#fc8181;margin:0.4rem 0;">False Positive (5× cost)</div>
        <div style="color:#94a3b8;font-size:0.875rem;line-height:1.6;">
            Blocking a genuine customer destroys Customer Lifetime Value, generates negative reviews,
            triggers chargebacks, and creates permanent brand damage. Far costlier than a missed fraud.
        </div>
    </div>
    """, unsafe_allow_html=True)
    cost_c2.markdown("""
    <div class="card" style="border-top:3px solid #68d391;">
        <div style="font-size:1.5rem;">💸</div>
        <div style="font-weight:700;color:#68d391;margin:0.4rem 0;">False Negative (1× cost)</div>
        <div style="color:#94a3b8;font-size:0.875rem;line-height:1.6;">
            A missed fraudulent return results in a small, bounded monetary loss that can be absorbed
            operationally. The model is tuned to aggressively protect genuine customers.
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# MAIN APP
# ──────────────────────────────────────────────────────────────

def main():
    render_header()

    # Check model availability
    if not render_model_status():
        st.stop()

    # Tab layout
    tab_analyze, tab_about = st.tabs(["🔍 Analyze Return", "📊 About & Benchmarks"])

    with tab_analyze:
        # Sidebar
        order_dict, fp_cost, fn_cost, run_flag = render_sidebar()

        # Load pipeline
        try:
            pipeline = load_pipeline()
        except FileNotFoundError as e:
            st.error(f"Model loading failed: {e}")
            st.stop()

        # Session state init
        if "last_result" not in st.session_state:
            st.session_state.last_result    = None
            st.session_state.last_order     = None
            st.session_state.last_fp_cost   = fp_cost
            st.session_state.history        = []
            st.session_state.reset_flag     = False

        # ── Reset button ──────────────────────────────────────
        if st.session_state.get("reset_flag"):
            st.session_state.last_result = None
            st.session_state.last_order  = None
            st.session_state.reset_flag  = False
            st.rerun()

        should_run = (
            run_flag or
            st.session_state.last_result is None or
            st.session_state.last_fp_cost != fp_cost
        )

        if should_run and order_dict:
            with st.spinner("🔍 Analyzing return request …"):
                try:
                    result = pipeline.predict(order_dict, fp_cost=fp_cost, fn_cost=fn_cost)
                    st.session_state.last_result  = result
                    st.session_state.last_order   = order_dict
                    st.session_state.last_fp_cost = fp_cost
                    # Append to history (keep last 20)
                    st.session_state.history.append({
                        "risk_score" : round(result.risk_score, 1),
                        "category"   : result.category,
                        "action"     : result.recommended_action,
                        "complaint"  : result.nlp_complaint_category,
                        "order"      : dict(order_dict),
                    })
                    if len(st.session_state.history) > 20:
                        st.session_state.history = st.session_state.history[-20:]
                except Exception as e:
                    st.error(f"Inference error: {e}")
                    st.stop()

        if st.session_state.last_result:
            render_result(st.session_state.last_result)

            # ── Action buttons row ────────────────────────────
            st.markdown("<br>", unsafe_allow_html=True)
            st.divider()
            st.markdown('<p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#63b3ed;margin-bottom:0.8rem;">⚡ Actions</p>', unsafe_allow_html=True)

            btn_c1, btn_c2, btn_c3, btn_c4 = st.columns(4)

            # — Export JSON ————————————————————————————————————
            with btn_c1:
                import json as _json
                result_r = st.session_state.last_result
                export_data = {
                    "risk_score"         : round(result_r.risk_score, 2),
                    "category"           : result_r.category,
                    "recommended_action" : result_r.recommended_action,
                    "behavior_prob"      : round(result_r.behavior_prob, 4),
                    "nlp_risk_score"     : round(result_r.nlp_risk_score, 4),
                    "nlp_complaint_cat"  : result_r.nlp_complaint_category,
                    "top_reasons"        : result_r.top_reasons,
                    "t_low"              : result_r.t_low,
                    "t_high"             : result_r.t_high,
                    "shap_values"        : dict(zip(
                        result_r.shap_feature_names or [],
                        [round(v, 5) for v in (result_r.shap_values or [])]
                    )),
                    "input_features"     : st.session_state.last_order,
                }
                st.download_button(
                    label="📥 Export JSON",
                    data=_json.dumps(export_data, indent=2, default=str),
                    file_name="fraud_triage_result.json",
                    mime="application/json",
                    use_container_width=True,
                    help="Download the full triage result as JSON",
                )

            # — Export CSV ————————————————————————————————————
            with btn_c2:
                result_r = st.session_state.last_result
                csv_row  = {
                    "risk_score"   : round(result_r.risk_score, 2),
                    "category"     : result_r.category,
                    "action"       : result_r.recommended_action,
                    "behavior_pct" : round(result_r.behavior_prob * 100, 1),
                    "nlp_risk_pct" : round(result_r.nlp_risk_score * 100, 1),
                    "complaint_cat": result_r.nlp_complaint_category,
                    "top_reason_1" : result_r.top_reasons[0] if result_r.top_reasons else "",
                }
                csv_df = pd.DataFrame([csv_row])
                st.download_button(
                    label="📄 Export CSV",
                    data=csv_df.to_csv(index=False),
                    file_name="fraud_triage_result.csv",
                    mime="text/csv",
                    use_container_width=True,
                    help="Download result as a CSV row",
                )

            # — Show History ——————————————————————————————————
            with btn_c3:
                if st.button("🕒 View History", use_container_width=True,
                             help=f"View last {len(st.session_state.history)} analyzed returns"):
                    st.session_state["show_history"] = not st.session_state.get("show_history", False)

            # — Reset ——————————————————————————————————————————
            with btn_c4:
                if st.button("🔄 Reset", use_container_width=True,
                             help="Clear current result and start fresh"):
                    st.session_state.reset_flag = True
                    st.rerun()

            # — History Panel ——————————————————————————————————
            if st.session_state.get("show_history") and st.session_state.history:
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown('<p style="font-size:0.75rem;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;color:#63b3ed;margin-bottom:0.8rem;">🕒 Analysis History (last 20)</p>', unsafe_allow_html=True)
                hist_df = pd.DataFrame([
                    {
                        "#"           : i + 1,
                        "Risk Score"  : h["risk_score"],
                        "Category"    : h["category"],
                        "Action"      : h["action"],
                        "Complaint"   : h["complaint"],
                    }
                    for i, h in enumerate(reversed(st.session_state.history))
                ])
                st.dataframe(
                    hist_df,
                    use_container_width=True,
                    height=min(35 * len(hist_df) + 40, 400),
                    hide_index=True,
                )
                # Download full history
                import json as _json2
                st.download_button(
                    "📥 Download Full History (JSON)",
                    data=_json2.dumps(st.session_state.history, indent=2, default=str),
                    file_name="fraud_triage_history.json",
                    mime="application/json",
                )

    with tab_about:
        render_about_tab()


if __name__ == "__main__":
    main()
