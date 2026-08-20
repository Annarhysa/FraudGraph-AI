"""
FraudGraph AI — Fraud Analyst Dashboard.

Run:
    streamlit run app.py

Every page shows analysis of the *signed-in user's uploaded data* — there
is no shared/demo dataset view. The underlying GraphSAGE model and its
calibration are still trained on the shared IBM TabFormer sample (see
src/fraudgraph/scoring.py, run once via the pipeline below); what changes
per user is which scored batch the dashboard is currently displaying.

Pipeline this depends on (run once, or let store.py build it on first load):
    python -m src.fraudgraph.data_prep
    python -m src.fraudgraph.graph_build
    python -m src.fraudgraph.anomaly
    python -m src.fraudgraph.gnn
    python -m src.fraudgraph.scoring
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent / "src"))

from fraudgraph import auth, batches as batch_store, store
from fraudgraph.ui import (
    alerts, case_management, components as ui, fraud_rings, investigation,
    model_monitoring, model_performance, network_explorer, overview,
    settings_about, transactions, upload_data,
)

st.set_page_config(page_title="FraudGraph AI", layout="wide", page_icon=ui.FAVICON_PATH)
ui.inject_css()

PAGES = {
    "Overview": overview,
    "Fraud Alerts": alerts,
    "Investigation": investigation,
    "Network Explorer": network_explorer,
    "Fraud Rings": fraud_rings,
    "Transactions": transactions,
    "My Uploads": upload_data,
    "Model Performance": model_performance,
    "Model Monitoring": model_monitoring,
    "Cases": case_management,
    "Settings / About": settings_about,
}

REQUIRED_FILES = [
    Path("data/processed/transactions_sample.parquet"),
    Path("data/processed/gnn_model.pt"),
]


def login_screen():
    ui.brand_header()
    st.caption("Sign in to score your own transactions and save the analysis to your account.")

    tab_login, tab_signup = st.tabs(["Log in", "Create account"])

    with tab_login:
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log in", type="primary")
            if submitted:
                user = auth.verify_user(username, password)
                if user is None:
                    st.error("Invalid username or password.")
                else:
                    token = auth.create_session(user["user_id"])
                    st.query_params["token"] = token
                    st.rerun()

    with tab_signup:
        with st.form("signup_form"):
            new_username = st.text_input("Choose a username")
            new_password = st.text_input("Choose a password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create account", type="primary")
            if submitted:
                if new_password != confirm_password:
                    st.error("Passwords don't match.")
                else:
                    try:
                        auth.create_user(new_username, new_password)
                        st.success("Account created — log in from the other tab.")
                    except ValueError as e:
                        st.error(str(e))

    st.caption(
        "Prototype-grade auth: passwords are salted/hashed (PBKDF2), but there's no rate "
        "limiting or password reset — don't reuse a real password here."
    )


def first_upload_screen(user: dict):
    ui.brand_header()
    st.subheader(f"Welcome, {user['username']}")
    st.write(
        "You haven't scored any transactions yet. Upload a CSV/Excel file to run it through "
        "the GraphSAGE model and rule engine — the dashboard will then show that analysis."
    )
    upload_data.render_format_help()
    batch_id = upload_data.render_upload_form(user, key_prefix="first")
    if batch_id:
        st.session_state["active_batch_id"] = batch_id
        st.rerun()


def chooser_screen(user: dict, history: list[dict]):
    ui.brand_header()
    st.subheader(f"Welcome back, {user['username']}")
    st.write("Pick up a previous analysis, or score a new file.")

    tab_existing, tab_new = st.tabs(["📂 Continue with an existing analysis", "📤 Upload a new file"])

    with tab_existing:
        for h in history:
            with st.container(border=True):
                c1, c2, c3 = st.columns([3, 1, 1])
                c1.write(f"**{h['filename']}**")
                c1.caption(f"{h['batch_id']} · scored {h['created_at'][:19]}")
                c2.metric("Transactions", h["n_transactions"])
                if c3.button("Open", key=f"chooser_open_{h['batch_id']}", type="primary"):
                    st.session_state["active_batch_id"] = h["batch_id"]
                    st.rerun()

    with tab_new:
        upload_data.render_format_help()
        batch_id = upload_data.render_upload_form(user, key_prefix="chooser_new")
        if batch_id:
            st.session_state["active_batch_id"] = batch_id
            st.rerun()


current_user = auth.get_user_from_token(st.query_params.get("token"))
if current_user is None:
    login_screen()
    st.stop()

missing = [f for f in REQUIRED_FILES if not f.exists()]
if missing:
    st.error(
        "Missing processed data/model files: " + ", ".join(str(m) for m in missing) +
        "\n\nRun the pipeline first:\n\n"
        "```\npython -m src.fraudgraph.data_prep\npython -m src.fraudgraph.graph_build\n"
        "python -m src.fraudgraph.anomaly\npython -m src.fraudgraph.gnn\n```"
    )
    st.stop()

st.session_state["current_user"] = current_user

user_history = batch_store.list_batches(current_user["user_id"])
valid_batch_ids = {h["batch_id"] for h in user_history}

if st.session_state.get("active_batch_id") not in valid_batch_ids:
    st.session_state.pop("active_batch_id", None)

if "active_batch_id" not in st.session_state:
    if not user_history:
        first_upload_screen(current_user)
    else:
        chooser_screen(current_user, user_history)
    st.stop()

# --- full dashboard ---
active_meta = next(h for h in user_history if h["batch_id"] == st.session_state["active_batch_id"])

with st.container(border=True):
    brand_col, user_col, logout_col = st.columns([5, 2, 1], vertical_alignment="center")
    with brand_col:
        ui.brand_header(heading="markdown")
    with user_col:
        st.markdown(f"👤 **{current_user['username']}**")
    with logout_col:
        if st.button("Log out", width="stretch"):
            auth.delete_session(st.query_params.get("token"))
            del st.query_params["token"]
            st.session_state.pop("active_batch_id", None)
            st.rerun()

    st.caption(f"Viewing **{active_meta['filename']}** ({active_meta['n_transactions']:,} txns)")

    nav_col, switch_col = st.columns([5, 3], vertical_alignment="bottom")
    with nav_col:
        default_page = st.session_state.get("nav_page", "Overview")
        page_name = st.selectbox(
            "Analysis", list(PAGES.keys()),
            index=list(PAGES.keys()).index(default_page),
        )
        st.session_state["nav_page"] = page_name
    with switch_col:
        if st.button("Switch analysis / new upload", width="stretch"):
            del st.session_state["active_batch_id"]
            st.rerun()

st.divider()

PAGES[page_name].render(store)

st.divider()
st.caption("v1 · research prototype, not a production fraud engine")
