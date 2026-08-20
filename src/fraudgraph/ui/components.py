"""Small shared UI helpers for a consistent enterprise-dashboard look."""
import base64
from functools import lru_cache
from pathlib import Path

import streamlit as st

FAVICON_PATH = "public/favicon.png"

# Brand palette
BRAND_PRIMARY = "#2563EB"
BRAND_PRIMARY_DARK = "#1E3A8A"
BRAND_ACCENT = "#60A5FA"
BRAND_BG_LIGHT = "#F8FAFC"
BRAND_BG_DARK = "#0F172A"

# Risk colors kept semantically distinct from the brand blue (LOW uses a
# neutral slate instead of blue so it doesn't read as "brand accent").
RISK_COLORS = {"LOW": "#64748B", "MEDIUM": "#D4A72C", "HIGH": "#E07B39", "CRITICAL": "#DC2626"}

GLOBAL_CSS = f"""
<style>
[data-testid="stMetricValue"] {{ font-size: 1.7rem; color: {BRAND_PRIMARY}; }}

.fg-badge {{
    display: inline-block; padding: 2px 10px; border-radius: 4px;
    font-size: 0.78rem; font-weight: 600; letter-spacing: 0.02em; color: white;
}}
.fg-section-title {{
    font-size: 0.95rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.05em; opacity: 0.85; margin-top: 0.5rem; margin-bottom: 0.3rem;
    color: {BRAND_PRIMARY_DARK};
}}
.fg-card {{
    border: 1px solid rgba(37,99,235,0.2); border-radius: 8px;
    padding: 0.9rem 1.1rem; margin-bottom: 0.6rem; background: rgba(37,99,235,0.05);
}}
.fg-brand {{
    display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.2rem;
}}
.fg-brand img {{ border-radius: 8px; display: block; }}
.fg-brand span {{ font-weight: 800; color: #000000; line-height: 1; }}

a {{ color: {BRAND_PRIMARY} !important; }}

/* Input/select boxes were nearly invisible against the white background —
   give them a visible themed border. Targets baseweb's own wrapper
   attributes (stable across Streamlit versions) rather than Streamlit's
   generated class names. */
div[data-baseweb="input"], div[data-baseweb="base-input"],
div[data-baseweb="select"] > div, div[data-baseweb="textarea"] {{
    border: 1.5px solid {BRAND_PRIMARY_DARK} !important;
    border-radius: 6px !important;
    background-color: #FFFFFF !important;
}}
div[data-baseweb="input"]:focus-within, div[data-baseweb="base-input"]:focus-within,
div[data-baseweb="select"] > div:focus-within, div[data-baseweb="textarea"]:focus-within {{
    border-color: {BRAND_PRIMARY} !important;
    box-shadow: 0 0 0 1px {BRAND_PRIMARY} !important;
}}
</style>
"""


def inject_css():
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


@lru_cache(maxsize=1)
def _favicon_base64() -> str:
    return base64.b64encode(Path(FAVICON_PATH).read_bytes()).decode()


def brand_header(heading: str = "title"):
    """Logo + 'FraudGraph AI' as a single flex row with a fixed, precise
    gap — heading='title' for a large header (login/upload/chooser
    screens), 'markdown' for the compact navbar version."""
    size = 40 if heading == "title" else 30
    font_size = "2.25rem" if heading == "title" else "1.35rem"
    st.markdown(
        f'<div class="fg-brand">'
        f'<img src="data:image/png;base64,{_favicon_base64()}" width="{size}" height="{size}"/>'
        f'<span style="font-size:{font_size}">FraudGraph AI</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def risk_badge(level: str) -> str:
    color = RISK_COLORS.get(level, "#888")
    return f'<span class="fg-badge" style="background:{color}">{level}</span>'

def risk_badge_html(level: str):
    st.markdown(risk_badge(level), unsafe_allow_html=True)


def section_title(text: str):
    st.markdown(f'<div class="fg-section-title">{text}</div>', unsafe_allow_html=True)


def progress_bar_text(label: str, pct: float):
    pct = float(pct)
    st.write(f"**{label}**")
    st.progress(min(max(pct / 100, 0.0), 1.0), text=f"{pct:.1f}%")
