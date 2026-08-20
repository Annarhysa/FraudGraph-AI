import streamlit as st

from ..ui import components as ui


def render(store):
    st.title("Settings / About")

    ui.section_title("FraudGraph AI")
    st.write("Explainable Financial Fraud Detection & Investigation")
    st.write("**Version:** v1 (research prototype)")
    st.write("**Model version:** GraphSAGE v1")

    st.divider()
    ui.section_title("Your active upload")
    active_df = store.active_df()
    if active_df is not None and not active_df.empty:
        st.write(f"**Transactions:** {len(active_df):,}")
        st.write(f"**Date range:** {active_df['txn_ts'].min()} to {active_df['txn_ts'].max()}")
        if bool(active_df["has_fraud_labels"].iloc[0]):
            st.write(f"**Fraud rate:** {active_df['is_fraud'].mean():.4%}")
        else:
            st.write("**Fraud rate:** N/A — no `Is Fraud?` column in this upload")

        active_data, active_meta = store.active_graph()
        st.write(f"**Graph nodes:** {active_data.num_nodes:,}  (cards, transactions, merchants)")
        st.write(f"**Graph edges:** {active_data.num_edges:,}")
    else:
        st.info("No active upload.")

    st.divider()
    ui.section_title("Model's training data (fixed, not your upload)")
    st.write(
        "The GraphSAGE model itself is trained once, offline, on a shared sample — it is **not** "
        "retrained on anything you upload (see §13 in the README for why that's safe: GraphSAGE "
        "generalizes to new graphs rather than memorizing fixed node embeddings)."
    )
    st.write("**Source:** IBM TabFormer credit card transactions (`data/data_transaction.csv`)")
    st.write("**Training sample:** 300 users + every user with a fraud transaction, 2019+ (~678K transactions)")

    st.divider()
    ui.section_title("Model architecture")
    st.write("2-layer GraphSAGE, hidden dim 32, trained with class-weighted BCE loss "
             "(pos_weight for the ~0.3% fraud base rate), temporal train/val/test split.")
    st.write("Displayed risk scores are Platt-calibrated probabilities, then min-max rescaled "
             "onto 0-100 for the analyst UI (see Model Performance page for the raw metrics — "
             "ROC-AUC/PR-AUC/precision@K are computed on the uncalibrated ranking and unaffected "
             "by this rescale, since it's monotonic).")

    st.divider()
    ui.section_title("Technical stack")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**Backend:** Python / FastAPI (`api/`)")
        st.write("**ML:** PyTorch / PyTorch Geometric (GraphSAGE)")
        st.write("**Graph:** NetworkX + PyG")
        st.write("**Explainability:** GNNExplainer + rule-based + graph-connectivity signals")
    with c2:
        st.write("**Data:** DuckDB / Parquet")
        st.write("**UI:** Streamlit")
        st.write("**Case/user/upload storage:** SQLite")
        st.write("**Clustering:** Louvain community detection (networkx)")

    st.divider()
    ui.section_title("Accounts & your uploaded data")
    st.write(
        "Every analyst has their own account (`fraudgraph.auth`) — passwords are salted and "
        "hashed with PBKDF2 (200,000 iterations), never stored in plaintext. Signing in issues "
        "a session token carried in the page URL (`?token=...`), which is what makes a login "
        "survive a browser refresh, not just an in-memory session."
    )
    st.write(
        "Anything scored on the **My Uploads** page is saved as a batch tied to your account "
        "(`fraudgraph.batches`) — metadata in SQLite, the scored transactions themselves in "
        "Parquet. Your upload history persists across refreshes, logouts, and app restarts, and "
        "each user only ever sees their own batches. Reports can be downloaded as CSV from any "
        "saved batch."
    )
    st.caption(
        "Prototype-grade auth: no rate limiting, no password reset flow, no email verification. "
        "Fine for a demo/local tool; don't reuse a real password here."
    )

    st.divider()
    ui.section_title("Limitations")
    st.markdown(
        "- This is a **research/prototype system**, not a production banking fraud engine.\n"
        "- Working sample is 300 users (+ all fraud-labeled users) restricted to 2019+, not the full 24.4M-row dataset.\n"
        "- Fraud labels in this sample only occur in the first ~70% of the time-sorted rows — a data artifact of the "
        "source dataset, not a model limitation, but it constrains where positives are available for evaluation.\n"
        "- GNNExplainer runs on a bounded 2-hop subgraph for interactivity, not the full graph.\n"
        "- Model monitoring reflects one static batch scoring pass, not a live production pipeline — there is no "
        "concept-drift detection or real request-level latency tracking.\n"
        "- Combined GNN+rule risk scores are intentionally **not** computed — the two are shown separately so no "
        "arbitrary weighting is hidden from the analyst."
    )
