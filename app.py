from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# =========================================================
# 1. PROJECT PATHS AND CONFIGURATION
# =========================================================

ROOT = Path(__file__).resolve().parent

DATA_PATH = ROOT / "data" / "battery_dataset.csv"
MODEL_PATH = ROOT / "models" / "battery_soh_model.joblib"
METRICS_PATH = ROOT / "reports" / "model_evaluation.json"

# Best-effort: force the light theme even if .streamlit/config.toml
# is missing (takes full effect after a page reload).
try:
    for _key, _val in {
        "theme.base": "light",
        "theme.primaryColor": "#0F766E",
        "theme.backgroundColor": "#F4F7F8",
        "theme.secondaryBackgroundColor": "#FFFFFF",
        "theme.textColor": "#14212B",
    }.items():
        st._config.set_option(_key, _val)
except Exception:
    pass

st.set_page_config(
    page_title="Battery Health Prediction System",
    page_icon="🔋",
    layout="wide"
)

FEATURES = [
    "Cycle_Count",
    "Avg_Temperature_C",
    "Avg_Cell_Voltage_V",
    "Avg_Current_A",
    "Depth_of_Discharge_Percent",
    "Charge_Rate_C",
]

FEATURE_LABELS = {
    "Cycle_Count": "Cycle count",
    "Avg_Temperature_C": "Average temperature (°C)",
    "Avg_Cell_Voltage_V": "Average cell voltage (V)",
    "Avg_Current_A": "Average current (A)",
    "Depth_of_Discharge_Percent": "Depth of discharge (%)",
    "Charge_Rate_C": "Charge rate (C-rate)",
}

# Input ranges used by the prediction form and CSV validation.
LIMITS = {
    "Cycle_Count": (0, 1500),
    "Avg_Temperature_C": (0.0, 55.0),
    "Avg_Cell_Voltage_V": (3.0, 4.25),
    "Avg_Current_A": (0.1, 4.0),
    "Depth_of_Discharge_Percent": (20, 100),
    "Charge_Rate_C": (0.2, 2.5),
}

DEFAULTS = {
    "Cycle_Count": 300,
    "Avg_Temperature_C": 28.0,
    "Avg_Cell_Voltage_V": 3.90,
    "Avg_Current_A": 1.4,
    "Depth_of_Discharge_Percent": 70,
    "Charge_Rate_C": 1.0,
}

PAGES = [
    "Dashboard",
    "Predict SOH",
    "Explore Dataset",
]

# SOH status bands (demonstration categories only).
GOOD_MIN = 90
MODERATE_MIN = 80

# Palette (kept in one place so charts and CSS stay consistent).
INK = "#14212B"
MUTED = "#5E6E78"
LINE = "#DCE3E6"
TEAL = "#0F766E"
TEAL_SOFT = "#7FBFB8"
GOOD = "#1F9D55"
MODERATE = "#D98E04"
DEGRADED = "#D1433A"


# =========================================================
# 2. STYLING
# =========================================================

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root {
    --ink: #14212B;
    --muted: #5E6E78;
    --line: #DCE3E6;
    --bg: #F4F7F8;
    --surface: #FFFFFF;
    --teal: #0F766E;
    --teal-soft: #E3F2F0;
    --good: #1F9D55;
    --moderate: #D98E04;
    --degraded: #D1433A;
}

.stApp,
.stApp p, .stApp label, .stApp li, .stApp td, .stApp th,
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp input, .stApp button, .stApp textarea {
    font-family: 'IBM Plex Sans', 'Segoe UI', system-ui, sans-serif;
}

.stApp { background: var(--bg); color: var(--ink); color-scheme: light; }

/* Force readable text whatever theme Streamlit / the browser is using. */
.stApp h1, .stApp h2, .stApp h3, .stApp h4,
.stApp p, .stApp li, .stApp label, .stApp legend,
.stApp [data-testid="stWidgetLabel"] p,
.stApp [data-testid="stMarkdownContainer"] p,
.stApp [data-testid="stRadio"] label p,
.stApp [data-testid="stCaptionContainer"] p,
.stApp [data-testid="stCaptionContainer"],
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span,
section[data-testid="stSidebar"] div[role="radiogroup"] label p {
    color: var(--ink);
}
.stApp [data-testid="stCaptionContainer"] p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: var(--muted);
}

/* ---------- Widgets: force light surfaces + dark text ---------- */

/* Text inputs, number inputs, select boxes (every nested layer) */
.stApp div[data-baseweb="input"],
.stApp div[data-baseweb="input"] *,
.stApp div[data-baseweb="base-input"],
.stApp div[data-baseweb="base-input"] *,
.stApp div[data-baseweb="select"],
.stApp div[data-baseweb="select"] * {
    background-color: #FFFFFF;
    color: var(--ink);
    -webkit-text-fill-color: var(--ink);
}
.stApp div[data-baseweb="input"],
.stApp div[data-baseweb="select"] > div {
    border: 1px solid var(--line);
    border-radius: 8px;
}
.stApp div[data-baseweb="select"] svg,
.stApp div[data-baseweb="input"] svg { fill: var(--muted); color: var(--muted); }
.stApp div[data-testid="stNumberInput"] button,
.stApp div[data-testid="stNumberInput"] button * {
    background-color: #F1F5F6; color: var(--ink);
}
.stApp div[data-testid="stNumberInput"] button:hover { background-color: var(--teal-soft); }

/* Dropdown menus open in a portal outside .stApp */
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] [role="option"],
div[data-baseweb="popover"] [role="listbox"] {
    background-color: #FFFFFF;
    color: #14212B;
}
div[data-baseweb="popover"] li *,
div[data-baseweb="popover"] [role="option"] * {
    background-color: transparent;
    color: #14212B;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] li[aria-selected="true"] {
    background-color: #E3F2F0;
}

/* Secondary buttons (download buttons, etc.) */
.stApp button[kind="secondary"],
.stApp button[kind="secondaryFormSubmit"],
.stApp div[data-testid="stDownloadButton"] button {
    background: #FFFFFF; border: 1px solid #B9C7CD; color: var(--ink);
}
.stApp button[kind="secondary"] *,
.stApp button[kind="secondaryFormSubmit"] *,
.stApp div[data-testid="stDownloadButton"] button * {
    color: var(--ink); background: transparent;
}
.stApp button[kind="secondary"]:hover,
.stApp button[kind="secondaryFormSubmit"]:hover,
.stApp div[data-testid="stDownloadButton"] button:hover {
    background: var(--teal-soft); border-color: var(--teal);
}

/* Primary button label must stay white (generic p rule would darken it) */
.stApp button[kind="primary"] *,
.stApp button[kind="primaryFormSubmit"] * { color: #FFFFFF; background: transparent; }

/* File uploader */
.stApp div[data-testid="stFileUploader"] section,
.stApp div[data-testid="stFileUploaderDropzone"] {
    background: #FFFFFF; color: var(--ink);
}
.stApp div[data-testid="stFileUploader"] section *,
.stApp div[data-testid="stFileUploaderDropzone"] *,
.stApp div[data-testid="stFileUploader"] small,
.stApp div[data-testid="stFileUploaderFile"] * { color: var(--ink); }
.stApp div[data-testid="stFileUploader"] section button {
    background: #FFFFFF; border: 1px solid #B9C7CD;
}
.stApp div[data-testid="stFileUploader"] section button:hover { background: var(--teal-soft); }

/* Expanders, alerts, sliders, JSON viewer */
.stApp div[data-testid="stExpander"] summary,
.stApp div[data-testid="stExpander"] summary *,
.stApp div[data-testid="stExpander"] details { color: var(--ink); background: transparent; }
.stApp div[data-testid="stAlert"] p,
.stApp div[data-testid="stAlert"] * { color: var(--ink); }
.stApp div[data-testid="stSlider"] [data-testid="stTickBarMin"],
.stApp div[data-testid="stSlider"] [data-testid="stTickBarMax"] { color: var(--muted); }
.stApp div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] + div,
.stApp div[data-testid="stSlider"] [data-testid="stThumbValue"] {
    background: transparent; color: var(--teal);
}
.stApp div[data-testid="stJson"], .stApp div[data-testid="stJson"] * {
    background-color: #FFFFFF; color: var(--ink);
}

header[data-testid="stHeader"] { background: transparent; }
footer { visibility: hidden; }

.block-container {
    padding-top: 2.2rem;
    padding-bottom: 3rem;
    max-width: 1180px;
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: var(--surface);
    border-right: 1px solid var(--line);
}
.brand {
    display: flex; align-items: center; gap: 12px;
    padding: 4px 0 18px 0;
    border-bottom: 1px solid var(--line);
    margin-bottom: 18px;
}
.brand-name { font-weight: 600; font-size: 1.02rem; line-height: 1.2; color: var(--ink); }
.brand-sub { font-size: 0.78rem; color: var(--muted); }
section[data-testid="stSidebar"] div[role="radiogroup"] { gap: 2px; }
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    padding: 8px 10px; border-radius: 8px;
}
section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: var(--teal-soft);
}

/* ---------- Page header ---------- */
.page-head { margin-bottom: 14px; }
.page-head h1 {
    font-size: 2rem; font-weight: 700; letter-spacing: -0.02em;
    color: var(--ink); margin: 0; padding: 0;
}
.stApp .page-head p { color: var(--muted); margin: 6px 0 0 0; font-size: 1rem; max-width: 70ch; }

.notice {
    border: 1px solid #F0DDB0; border-left: 4px solid var(--moderate);
    background: #FFF9EA; color: #5B4510;
    border-radius: 8px; padding: 10px 14px; margin: 14px 0 26px 0;
    font-size: 0.88rem; line-height: 1.5;
}

.section-title {
    font-size: 1.1rem; font-weight: 600; color: var(--ink);
    margin: 28px 0 10px 0;
}
.section-sub { color: var(--muted); font-size: 0.9rem; margin: -6px 0 12px 0; }

/* ---------- Cards ---------- */
.kpi {
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 12px; padding: 16px 18px; height: 100%;
}
.kpi-label { color: var(--muted); font-size: 0.82rem; }
.kpi-value {
    font-size: 1.55rem; font-weight: 600; color: var(--ink);
    margin-top: 2px; font-variant-numeric: tabular-nums; letter-spacing: -0.01em;
}
.kpi-sub { color: var(--muted); font-size: 0.78rem; margin-top: 2px; }

.card {
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 12px; padding: 20px 22px;
}
.group-label { font-weight: 600; color: var(--ink); margin: 2px 0 6px 0; font-size: 0.95rem; }

/* ---------- Prediction result ---------- */
.res-top { display: flex; justify-content: space-between; align-items: flex-start; gap: 10px; }
.res-label { color: var(--muted); font-size: 0.85rem; }
.res-value {
    font-size: 3.1rem; font-weight: 700; line-height: 1.05;
    letter-spacing: -0.03em; font-variant-numeric: tabular-nums;
}
.res-value span { font-size: 1.5rem; font-weight: 500; margin-left: 2px; color: var(--muted); }
.pill {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: 0.82rem; font-weight: 600; white-space: nowrap;
}
.pill-good { background: #E4F5EB; color: #17703E; }
.pill-moderate { background: #FCF1D9; color: #8A5A00; }
.pill-degraded { background: #FBE5E3; color: #9E2A22; }
.pill-idle { background: #EDF1F3; color: var(--muted); }
.gauge { margin: 14px 0 6px 0; }
.gauge svg { width: 100%; height: auto; display: block; }
.res-note { color: var(--muted); font-size: 0.84rem; line-height: 1.55; margin-top: 10px; }

.inputs-used { margin-top: 14px; border-top: 1px solid var(--line); padding-top: 10px; }
.inputs-used .row {
    display: flex; justify-content: space-between;
    padding: 5px 0; font-size: 0.88rem; border-bottom: 1px dashed #E6ECEE;
}
.inputs-used .row:last-child { border-bottom: none; }
.inputs-used .k { color: var(--muted); }
.inputs-used .v { font-weight: 500; font-variant-numeric: tabular-nums; }

/* ---------- Streamlit widgets ---------- */
div[data-testid="stForm"] {
    background: var(--surface); border: 1px solid var(--line);
    border-radius: 12px; padding: 20px 22px;
}
.stApp button[kind="primary"], .stApp button[kind="primaryFormSubmit"] {
    background: var(--teal); border-color: var(--teal); color: #fff; font-weight: 600;
}
.stApp button[kind="primary"]:hover, .stApp button[kind="primaryFormSubmit"]:hover {
    background: #0B5F59; border-color: #0B5F59; color: #fff;
}
div[data-testid="stFileUploader"] section {
    background: var(--surface); border: 1.5px dashed #B9C7CD; border-radius: 12px;
}
div[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 10px; }
div[data-testid="stExpander"] {
    background: var(--surface); border: 1px solid var(--line); border-radius: 10px;
}
</style>
"""


def html(markup):
    """
    Render HTML safely inside Streamlit markdown.
    Lines are stripped and joined so indentation or blank lines
    are never interpreted as a Markdown code block.
    """
    flat = "".join(line.strip() for line in markup.splitlines())
    st.markdown(flat, unsafe_allow_html=True)


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title, subtitle):
    html(
        f"""
        <div class="page-head">
            <h1>{title}</h1>
            <p>{subtitle}</p>
        </div>
        """
    )


def safety_notice():
    html(
        """
        <div class="notice">
            <strong>Academic prototype.</strong>
            The model was trained on simulated data and its predictions
            have not been validated on real batteries. Do not use these
            results for battery safety decisions.
        </div>
        """
    )


def section_title(text, sub=None):
    html(f'<div class="section-title">{text}</div>')

    if sub:
        html(f'<div class="section-sub">{sub}</div>')


def kpi_card(label, value, sub=""):
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""

    html(
        f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
        """
    )


# ---------- Matplotlib styling ----------

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#E8EDEF",
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


def new_figure(width=6.2, height=3.7):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor("white")
    return fig, ax


def show_figure(fig):
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def add_trend_line(ax, x, y, bins=12):
    """
    Overlay the median SOH across equal-sized bins of x.
    """
    frame = pd.DataFrame({"x": x, "y": y}).dropna()

    if frame["x"].nunique() < 8:
        return

    frame["bin"] = pd.qcut(frame["x"], bins, duplicates="drop")

    grouped = frame.groupby("bin", observed=True).agg(
        x=("x", "median"),
        y=("y", "median"),
    )

    ax.plot(
        grouped["x"],
        grouped["y"],
        color=INK,
        linewidth=2,
        label="Median trend",
    )

    ax.legend(frameon=False, loc="best")


# ---------- SOH status and battery gauge ----------

def soh_status(pred):
    """
    Return (label, css class, hex color) for a predicted SOH value.
    """
    if pred >= GOOD_MIN:
        return "Good", "good", GOOD

    if pred >= MODERATE_MIN:
        return "Moderate", "moderate", MODERATE

    return "Degraded", "degraded", DEGRADED


def battery_gauge_svg(pred=None):
    """
    Horizontal battery whose fill level equals the predicted SOH.
    Tick marks show the 80% and 90% status thresholds.
    """
    inner_x, inner_w = 14, 260

    if pred is None:
        fill_w = 0
        color = "#C8D2D7"
    else:
        fill_w = inner_w * max(0, min(100, pred)) / 100
        color = soh_status(pred)[2]

    fill = (
        f'<rect x="{inner_x}" y="30" width="{fill_w:.1f}" height="76" '
        f'rx="9" fill="{color}"/>'
        if fill_w > 0
        else ""
    )

    ticks = ""

    for threshold in (MODERATE_MIN, GOOD_MIN):
        x = inner_x + inner_w * threshold / 100

        ticks += (
            f'<line x1="{x:.1f}" y1="118" x2="{x:.1f}" y2="128" '
            f'stroke="{MUTED}" stroke-width="1.5"/>'
            f'<text x="{x:.1f}" y="143" text-anchor="middle" '
            f'font-size="11" fill="{MUTED}">{threshold}%</text>'
        )

    return (
        '<svg viewBox="0 0 320 150" xmlns="http://www.w3.org/2000/svg" '
        'role="img" aria-label="Battery state of health gauge">'
        f'<rect x="4" y="20" width="280" height="96" rx="16" '
        f'fill="#F4F7F8" stroke="{INK}" stroke-width="4"/>'
        f'<rect x="288" y="48" width="18" height="40" rx="6" fill="{INK}"/>'
        f'{fill}{ticks}</svg>'
    )


def render_result_panel(result, source):
    """
    Result card: SOH value, status pill, battery gauge, inputs used.
    `result` is the stored prediction dict, or None before the first run.
    """
    if result is None:
        value_html = '<div class="res-value">&ndash;<span>%</span></div>'
        pill_html = '<span class="pill pill-idle">No prediction yet</span>'
        gauge = battery_gauge_svg(None)
        note = (
            "Enter battery parameters and select "
            "<strong>Predict battery SOH</strong> to see the estimate."
        )
        inputs_html = ""
    else:
        pred = result["Predicted_SOH_Percent"]
        label, css, _ = soh_status(pred)

        value_html = f'<div class="res-value">{pred:.1f}<span>%</span></div>'
        pill_html = f'<span class="pill pill-{css}">{label}</span>'
        gauge = battery_gauge_svg(pred)
        note = (
            "SOH (state of health) is the estimated remaining capacity "
            "relative to reference capacity. The status is a demonstration "
            "category, not a verified battery diagnosis."
        )

        rows = "".join(
            f'<div class="row"><span class="k">{FEATURE_LABELS[f]}</span>'
            f'<span class="v">{result[f]:g}</span></div>'
            for f in FEATURES
        )

        inputs_html = (
            '<div class="inputs-used">'
            f'<div class="group-label">Inputs used ({source})</div>'
            f"{rows}</div>"
        )

    html(
        f"""
        <div class="card">
            <div class="res-top">
                <div>
                    <div class="res-label">Predicted state of health</div>
                    {value_html}
                </div>
                {pill_html}
            </div>
            <div class="gauge">{gauge}</div>
            <div class="res-note">{note}</div>
            {inputs_html}
        </div>
        """
    )


# =========================================================
# 3. LOAD DATA, MODEL AND METRICS
# =========================================================

@st.cache_data
def load_data():
    if not DATA_PATH.exists():
        return pd.DataFrame()

    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    if METRICS_PATH.exists():
        return json.loads(
            METRICS_PATH.read_text(encoding="utf-8")
        )

    return {}


data = load_data()
bundle = load_model()
metrics = load_metrics()


def flatten_metrics(obj, prefix=""):
    """
    Flatten a nested metrics dict into {"name": number} pairs.
    """
    flat = {}

    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix} {key}".strip()
            flat.update(flatten_metrics(value, name))

    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        flat[prefix] = obj

    return flat


def find_metric(flat, candidates):
    """
    Find the first metric whose name contains one of the candidates.
    """
    for name, value in flat.items():
        lowered = name.lower().replace("-", "_")

        if any(c in lowered for c in candidates):
            return name, value

    return None


# =========================================================
# 4. CSV VALIDATION AND FEATURE MAPPING
# =========================================================

def normalize_columns(df):
    """
    Normalize CSV column names to support minor differences
    in capitalization and whitespace.
    """
    df = df.copy()

    df.columns = [
        str(col).strip()
        for col in df.columns
    ]

    aliases = {
        "cycle_count": "Cycle_Count",
        "avg_temperature_c": "Avg_Temperature_C",
        "avg_cell_voltage_v": "Avg_Cell_Voltage_V",
        "avg_current_a": "Avg_Current_A",
        "depth_of_discharge_percent":
            "Depth_of_Discharge_Percent",
        "charge_rate_c": "Charge_Rate_C",
    }

    rename_map = {}

    for col in df.columns:
        normalized = col.lower().replace(" ", "_")

        if normalized in aliases:
            rename_map[col] = aliases[normalized]

    return df.rename(columns=rename_map)


def read_uploaded_csv(uploaded_file):
    """
    Read and validate an uploaded battery-profile CSV.
    """
    try:
        # Streamlit reruns reuse the same file buffer, so rewind first.
        uploaded_file.seek(0)

        raw = pd.read_csv(uploaded_file)

        if raw.empty:
            raise ValueError("The uploaded CSV contains no data rows.")

        raw = normalize_columns(raw)

        missing = [
            feature
            for feature in FEATURES
            if feature not in raw.columns
        ]

        if missing:
            raise ValueError(
                "Missing required columns: "
                + ", ".join(missing)
            )

        # Convert required input columns to numeric values.
        for feature in FEATURES:
            raw[feature] = pd.to_numeric(
                raw[feature],
                errors="coerce"
            )

        if raw[FEATURES].isna().any().any():
            raise ValueError(
                "Some required input values are empty or non-numeric. "
                "Please correct the CSV and upload it again."
            )

        # Validate all uploaded rows against the form's input ranges.
        for feature in FEATURES:
            low, high = LIMITS[feature]

            invalid = (
                (raw[feature] < low)
                | (raw[feature] > high)
            )

            if invalid.any():
                raise ValueError(
                    f"{feature} must be between {low} and {high}. "
                    "Please correct the CSV."
                )

        return raw

    except Exception as exc:
        raise ValueError(str(exc)) from exc


# =========================================================
# 5. SIDEBAR + GLOBAL HEADER
# =========================================================

inject_css()

with st.sidebar:
    html(
        """
        <div class="brand">
            <svg width="36" height="36" viewBox="0 0 36 36"
                 xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                <rect x="2" y="9" width="28" height="18" rx="5"
                      fill="none" stroke="#14212B" stroke-width="2.5"/>
                <rect x="31" y="14" width="3.5" height="8" rx="1.5"
                      fill="#14212B"/>
                <rect x="6" y="13" width="17" height="10" rx="2.5"
                      fill="#0F766E"/>
            </svg>
            <div>
                <div class="brand-name">Battery Health</div>
                <div class="brand-sub">SOH prediction system</div>
            </div>
        </div>
        """
    )

    page = st.radio(
        "Navigation",
        PAGES,
        label_visibility="collapsed"
    )

    st.markdown("---")

    st.caption(
        "Synthetic-data educational prototype. "
        "Not for real battery safety decisions."
    )


# =========================================================
# 6. DASHBOARD
# =========================================================

if page == "Dashboard":

    page_header(
        "Battery Health Prediction",
        "Estimate a battery's state of health (SOH) from its usage "
        "profile with a machine-learning model."
    )

    safety_notice()

    flat_metrics = flatten_metrics(metrics)
    r2 = find_metric(flat_metrics, ["r2", "r_squared"])

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        kpi_card(
            "Training records",
            f"{len(data):,}",
            "Synthetic dataset"
        )

    with c2:
        kpi_card(
            "Model",
            "Random Forest",
            "Regressor"
        )

    with c3:
        kpi_card(
            "Prediction target",
            "SOH (%)",
            "State of health"
        )

    with c4:
        if r2 is not None:
            kpi_card(
                "Model R²",
                f"{r2[1]:.3f}",
                "On synthetic data"
            )
        else:
            kpi_card(
                "Input features",
                f"{len(FEATURES)}",
                "Usage and operating conditions"
            )

    if data.empty:
        st.info(
            "The training dataset was not found. "
            "Check that data/battery_dataset.csv exists."
        )

    elif "SOH_Percent" in data.columns:

        section_title(
            "Dataset overview",
            "How state of health is distributed in the training data "
            "and how it changes as batteries age."
        )

        left, right = st.columns(2)

        with left:
            fig, ax = new_figure()

            ax.hist(
                data["SOH_Percent"],
                bins=28,
                color=TEAL,
                edgecolor="white",
                linewidth=0.6,
            )

            for threshold in (MODERATE_MIN, GOOD_MIN):
                ax.axvline(
                    threshold,
                    color=MUTED,
                    linestyle="--",
                    linewidth=1,
                )

            ax.set_xlabel("SOH (%)")
            ax.set_ylabel("Number of records")
            ax.set_title("SOH distribution")

            show_figure(fig)

        with right:
            if "Cycle_Count" in data.columns:

                sample = data.sample(
                    min(250, len(data)),
                    random_state=42
                ).sort_values("Cycle_Count")

                fig, ax = new_figure()

                ax.scatter(
                    sample["Cycle_Count"],
                    sample["SOH_Percent"],
                    alpha=0.45,
                    color=TEAL,
                    s=18,
                    edgecolors="none",
                )

                add_trend_line(
                    ax,
                    data["Cycle_Count"],
                    data["SOH_Percent"],
                )

                ax.set_xlabel("Cycle count")
                ax.set_ylabel("SOH (%)")
                ax.set_title("SOH vs cycle count")

                show_figure(fig)

    section_title(
        "Model inputs",
        "The six values the model uses, with the accepted range "
        "for each."
    )

    inputs_table = pd.DataFrame(
        [
            {
                "Input": FEATURE_LABELS[feature],
                "Column name": feature,
                "Accepted range":
                    f"{LIMITS[feature][0]} to {LIMITS[feature][1]}",
            }
            for feature in FEATURES
        ]
    )

    st.dataframe(
        inputs_table,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# 7. PREDICT SOH + CSV UPLOAD
# =========================================================

elif page == "Predict SOH":

    page_header(
        "Predict battery SOH",
        "Enter a battery's usage profile, or upload a CSV and pick "
        "a profile to fill the form automatically."
    )

    safety_notice()

    # Make sure every form widget has a starting value.
    for feature in FEATURES:
        st.session_state.setdefault(
            f"input_{feature}",
            DEFAULTS[feature]
        )

    form_col, result_col = st.columns([3, 2], gap="large")

    # =====================================================
    # LEFT: CSV UPLOAD + INPUT FORM
    # =====================================================

    with form_col:

        # -------------------------------------------------
        # 7A. CSV UPLOAD
        # -------------------------------------------------

        section_title("Load a profile from CSV (optional)")

        uploaded_file = st.file_uploader(
            "Choose a battery CSV file",
            type=["csv"],
            key="battery_csv_upload",
            label_visibility="collapsed",
            help=(
                "The CSV must contain all six required input columns. "
                "It may contain one row or multiple battery profiles."
            )
        )

        uploaded_data = None

        if uploaded_file is not None:

            try:
                uploaded_data = read_uploaded_csv(uploaded_file)

                st.success(
                    f"CSV loaded: {len(uploaded_data)} "
                    "battery profile(s) found."
                )

                with st.expander("Preview uploaded CSV"):
                    st.dataframe(
                        uploaded_data,
                        use_container_width=True
                    )

                selected_row = st.selectbox(
                    "Select a battery profile",
                    options=list(range(len(uploaded_data))),
                    format_func=lambda index: (
                        f"Profile {index + 1} "
                        f"(CSV row {index + 2})"
                    ),
                    key="selected_csv_row"
                )

                profile = uploaded_data.iloc[selected_row]

                # Fill the form only when the file or the selected row
                # changes. Otherwise values edited in the form would be
                # overwritten by the CSV on every submit.
                file_id = getattr(uploaded_file, "file_id", None) or (
                    f"{uploaded_file.name}:{uploaded_file.size}"
                )

                signature = (file_id, selected_row)

                if st.session_state.get("loaded_profile_sig") != signature:

                    for feature in FEATURES:

                        low, high = LIMITS[feature]

                        value = float(profile[feature])

                        if feature in [
                            "Cycle_Count",
                            "Depth_of_Discharge_Percent",
                        ]:
                            value = int(value)

                        value = max(low, min(high, value))

                        st.session_state[
                            f"input_{feature}"
                        ] = value

                    st.session_state["loaded_profile_sig"] = signature

                st.caption(
                    f"Profile {selected_row + 1} is loaded into the "
                    "form below. You can still edit any value."
                )

            except Exception as exc:

                uploaded_data = None

                st.error(
                    f"Could not read the uploaded CSV: {exc}"
                )

                st.info(
                    "Check the required column names and value ranges, "
                    "then upload the corrected file."
                )

        else:

            st.caption(
                "No CSV uploaded. You can enter the values manually."
            )

            # Clear the previous CSV selection when the uploader is empty.
            st.session_state.pop("selected_csv_row", None)
            st.session_state.pop("loaded_profile_sig", None)

        template = pd.DataFrame([DEFAULTS])

        st.download_button(
            "Download CSV template",
            data=template.to_csv(index=False),
            file_name="battery_profile_template.csv",
            mime="text/csv",
            key="download_csv_template",
            help="One row per battery profile."
        )

        # -------------------------------------------------
        # 7B. MANUAL INPUT FORM
        # -------------------------------------------------

        section_title("Battery parameters")

        if bundle is None:

            st.error(
                "Trained model not found. "
                "Run train_model.py first, then refresh the app."
            )

        else:

            with st.form("prediction_form"):

                left, right = st.columns(2)

                with left:

                    html('<div class="group-label">Usage history</div>')

                    cycle = st.number_input(
                        "Cycle count",
                        min_value=0,
                        max_value=1500,
                        step=10,
                        key="input_Cycle_Count",
                        help="Accepted range: 0 to 1500",
                    )

                    dod = st.slider(
                        "Depth of discharge (%)",
                        min_value=20,
                        max_value=100,
                        step=1,
                        key="input_Depth_of_Discharge_Percent",
                    )

                    rate = st.slider(
                        "Charge rate (C-rate)",
                        min_value=0.2,
                        max_value=2.5,
                        step=0.1,
                        key="input_Charge_Rate_C",
                    )

                with right:

                    html(
                        '<div class="group-label">'
                        "Operating conditions</div>"
                    )

                    temp = st.number_input(
                        "Average temperature (°C)",
                        min_value=0.0,
                        max_value=55.0,
                        step=0.5,
                        key="input_Avg_Temperature_C",
                        help="Accepted range: 0 to 55 °C",
                    )

                    voltage = st.number_input(
                        "Average cell voltage (V)",
                        min_value=3.0,
                        max_value=4.25,
                        step=0.01,
                        format="%.3f",
                        key="input_Avg_Cell_Voltage_V",
                        help="Accepted range: 3.0 to 4.25 V",
                    )

                    current = st.number_input(
                        "Average current (A)",
                        min_value=0.1,
                        max_value=4.0,
                        step=0.1,
                        key="input_Avg_Current_A",
                        help="Accepted range: 0.1 to 4.0 A",
                    )

                submitted = st.form_submit_button(
                    "Predict battery SOH",
                    type="primary",
                    use_container_width=True
                )

            # ---------------------------------------------
            # 7C. PREDICTION
            # ---------------------------------------------

            if submitted:

                input_values = {
                    "Cycle_Count": cycle,
                    "Avg_Temperature_C": temp,
                    "Avg_Cell_Voltage_V": voltage,
                    "Avg_Current_A": current,
                    "Depth_of_Discharge_Percent": dod,
                    "Charge_Rate_C": rate,
                }

                try:

                    # Preserve the feature order used during training.
                    model_features = bundle["features"]

                    row = pd.DataFrame(
                        [input_values]
                    ).reindex(
                        columns=model_features
                    )

                    pred = float(
                        bundle["model"].predict(row)[0]
                    )

                    pred = float(
                        np.clip(pred, 50, 100)
                    )

                    st.session_state["last_prediction"] = {
                        **input_values,
                        "Predicted_SOH_Percent": round(pred, 2),
                    }

                    st.session_state[
                        "last_prediction_source"
                    ] = (
                        "CSV upload"
                        if uploaded_file is not None
                        and uploaded_data is not None
                        else "Manual input"
                    )

                except Exception as exc:

                    st.error(
                        f"Prediction failed: {exc}"
                    )

    # =====================================================
    # RIGHT: RESULT PANEL + DOWNLOAD
    # =====================================================

    with result_col:

        section_title("Result")

        last_prediction = st.session_state.get("last_prediction")

        source = st.session_state.get(
            "last_prediction_source",
            "Unknown"
        )

        render_result_panel(last_prediction, source)

        if last_prediction is not None:

            prediction_csv = pd.DataFrame(
                [last_prediction]
            ).to_csv(index=False)

            st.download_button(
                "Download prediction as CSV",
                data=prediction_csv,
                file_name="battery_soh_prediction.csv",
                mime="text/csv",
                key="download_latest_prediction",
                use_container_width=True,
            )


# =========================================================
# 8. EXPLORE DATASET
# =========================================================

elif page == "Explore Dataset":

    page_header(
        "Explore dataset",
        "Compare any input with SOH in the synthetic data used to "
        "train this prototype."
    )

    safety_notice()

    if data.empty:

        st.error(
            "Dataset not found. Check data/battery_dataset.csv."
        )

    elif "SOH_Percent" not in data.columns:

        st.error(
            "The dataset does not contain the SOH_Percent column."
        )

    else:

        available_features = [
            feature
            for feature in FEATURES
            if feature in data.columns
        ]

        if available_features:

            left, right = st.columns(2)

            with left:

                selected = st.selectbox(
                    "Choose a feature to compare with SOH",
                    available_features,
                    format_func=lambda f: FEATURE_LABELS.get(f, f)
                )

            with right:

                rows_to_show = st.selectbox(
                    "Preview rows",
                    [10, 25, 50, 100],
                    index=1
                )

            correlation = data[selected].corr(data["SOH_Percent"])

            k1, k2, k3 = st.columns(3)

            with k1:
                kpi_card(
                    "Records",
                    f"{len(data):,}",
                    "In the full dataset"
                )

            with k2:
                kpi_card(
                    "Mean SOH",
                    f"{data['SOH_Percent'].mean():.1f}%",
                    f"Range {data['SOH_Percent'].min():.1f}% "
                    f"to {data['SOH_Percent'].max():.1f}%"
                )

            with k3:
                kpi_card(
                    "Correlation with SOH",
                    f"{correlation:+.2f}",
                    FEATURE_LABELS.get(selected, selected)
                )

            sample = data.sample(
                min(600, len(data)),
                random_state=7
            )

            fig, ax = new_figure(width=9, height=4)

            ax.scatter(
                sample[selected],
                sample["SOH_Percent"],
                alpha=0.4,
                color=TEAL,
                s=18,
                edgecolors="none",
            )

            add_trend_line(
                ax,
                data[selected],
                data["SOH_Percent"],
            )

            label = FEATURE_LABELS.get(selected, selected)

            ax.set_xlabel(label)
            ax.set_ylabel("SOH (%)")
            ax.set_title(f"{label} vs SOH")

            section_title("Relationship with SOH")
            show_figure(fig)

        section_title("Dataset preview")

        st.dataframe(
            data.head(rows_to_show if available_features else 25),
            use_container_width=True,
            hide_index=True
        )

        st.download_button(
            "Download full synthetic dataset",
            data=data.to_csv(index=False),
            file_name="battery_dataset.csv",
            mime="text/csv",
        )
