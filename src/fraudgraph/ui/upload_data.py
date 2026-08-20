import pandas as pd
import streamlit as st

from .. import batches as batch_store
from .. import explainability
from .. import gnn
from .. import netviz
from .. import scoring
from ..ui import components as ui

TEMPLATE_CSV = (
    "User,Card,Year,Month,Day,Time,Amount,Use Chip,Merchant Name,"
    "Merchant City,Merchant State,Zip,MCC,Errors?,Is Fraud?\n"
    "0,0,2019,3,4,06:21,$134.09,Swipe Transaction,3527213246127876953,"
    "La Verne,CA,91750.0,5300,,No\n"
    "0,0,2019,3,4,06:42,$38.48,Swipe Transaction,-727612092139916043,"
    "Monterey Park,CA,91754.0,5411,,No\n"
)

REPORT_COLUMNS = [
    "transaction_id", "risk_level", "gnn_score_pct", "rule_score_pct", "amount",
    "card_key", "merchant_id", "txn_ts", "is_fraud",
    "merchant_high_risk_cards", "card_shared_suspicious_accounts",
]


def _read_uploaded(file) -> pd.DataFrame:
    name = file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(file)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)
    raise ValueError("Unsupported file type — please upload a .csv or .xlsx/.xls file.")


def render_format_help():
    st.write(
        "CSV or Excel with these columns (same schema as the IBM TabFormer dataset "
        "this project was built on):"
    )
    st.code(
        "User, Card, Year, Month, Day, Time, Amount, Use Chip, Merchant Name, "
        "Merchant City, Merchant State, Zip, MCC\n"
        "Optional: Errors?, Is Fraud? (Yes/No — enables ground-truth comparison)",
        language=None,
    )
    st.download_button(
        "Download template CSV", TEMPLATE_CSV, file_name="fraudgraph_upload_template.csv",
        key="template_dl_btn",
    )


def render_upload_form(user: dict, key_prefix: str = "upload") -> str | None:
    """Renders the file-uploader + scoring flow. Returns the new batch_id
    once a file has been scored and saved, else None."""
    uploaded = st.file_uploader(
        "Upload transactions (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        key=f"{key_prefix}_file",
    )
    if uploaded is None:
        return None

    try:
        raw_df = _read_uploaded(uploaded)
    except Exception as e:
        st.error(f"Could not read file: {e}")
        return None

    st.caption(f"{len(raw_df):,} rows detected in **{uploaded.name}**.")
    if not st.button("Score these transactions", type="primary", key=f"{key_prefix}_score_btn"):
        return None

    status = st.status(f"Scoring {len(raw_df):,} transactions...", expanded=True)
    try:
        status.write("Parsing file and computing rule-based signals...")
        status.write("Building the transaction graph...")
        status.write("Running GraphSAGE inference (using the pretrained model, no retraining)...")
        scored_df, _data_obj, _meta = scoring.score_uploaded(raw_df)
        status.write("Computing network signals within this batch...")
        batch_id = batch_store.save_batch(user["user_id"], uploaded.name, scored_df)
        status.update(label=f"Scored {len(scored_df):,} transactions.", state="complete", expanded=False)
    except (ValueError, RuntimeError) as e:
        status.update(label="Scoring failed", state="error")
        st.error(str(e))
        return None
    except Exception as e:
        status.update(label="Scoring failed", state="error")
        st.error(f"Scoring failed: {e}")
        return None

    return batch_id


def render(store):
    st.title("My Uploads")
    st.caption(
        "Score a new file, or switch which of your saved analyses the dashboard is showing. "
        "Every upload is saved to your account and stays available after a refresh or new session."
    )

    user = st.session_state["current_user"]

    with st.expander("Upload a new file", expanded=False):
        render_format_help()
        new_batch_id = render_upload_form(user, key_prefix="myuploads")
        if new_batch_id:
            st.session_state["active_batch_id"] = new_batch_id
            st.rerun()

    st.divider()
    ui.section_title("Your upload history")
    history = batch_store.list_batches(user["user_id"])
    if not history:
        st.info("No uploads yet — score a file above to get started.")
        return

    active_id = store.active_batch_id()
    for h in history:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([3, 1, 1, 1])
            with c1:
                label = f"**{h['filename']}**" + ("  🟢 *currently active*" if h["batch_id"] == active_id else "")
                st.write(label)
                st.caption(f"{h['batch_id']} · scored {h['created_at'][:19]}")
            c2.metric("Transactions", h["n_transactions"])
            c3.metric("High risk+", h["n_high_risk"])
            if c4.button("View this", key=f"view_{h['batch_id']}", disabled=h["batch_id"] == active_id):
                st.session_state["active_batch_id"] = h["batch_id"]
                st.rerun()

            d1, d2 = st.columns([1, 5])
            df = batch_store.load_batch(h["batch_id"])
            has_labels = bool(df["has_fraud_labels"].iloc[0]) if len(df) else False
            report_cols = REPORT_COLUMNS if has_labels else [c for c in REPORT_COLUMNS if c != "is_fraud"]
            d1.download_button(
                "Download report", df[report_cols].to_csv(index=False),
                file_name=f"fraudgraph_report_{h['batch_id']}.csv", mime="text/csv",
                key=f"dl_{h['batch_id']}",
            )
            if d2.button("Delete", key=f"del_{h['batch_id']}"):
                batch_store.delete_batch(h["batch_id"], user["user_id"])
                if active_id == h["batch_id"]:
                    del st.session_state["active_batch_id"]
                st.rerun()
