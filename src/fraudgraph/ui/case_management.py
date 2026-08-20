import streamlit as st

from .. import cases as case_store
from ..ui import components as ui

PRIORITIES = case_store.PRIORITIES
STATUSES = case_store.STATUSES


def render(store):
    st.title("Cases")
    st.caption("Investigation cases for your active upload. Backed by SQLite (data/processed/fraudgraph.db).")

    df = store.active_df()
    if df is None or df.empty:
        st.info("No transactions in the active upload.")
        return

    txn_ids = set(df["transaction_id"])
    all_cases = [c for c in case_store.list_cases() if c["transaction_id"] in txn_ids]

    ui.section_title("Open a new case")
    with st.form("new_case"):
        c1, c2 = st.columns(2)
        txn_id = c1.selectbox("Transaction", df.sort_values("gnn_score_pct", ascending=False)["transaction_id"].head(500).tolist())
        priority = c2.selectbox("Priority", PRIORITIES, index=2)
        notes = st.text_area("Notes", placeholder="e.g. Card connected to 3 previously flagged merchants. Amount is 4.2x historical average.")
        submitted = st.form_submit_button("Create Case")
        if submitted:
            case_id = case_store.create_case(txn_id, priority=priority, notes=notes)
            st.success(f"Created {case_id}")
            st.rerun()

    st.divider()
    ui.section_title(f"All cases ({len(all_cases)})")

    if not all_cases:
        st.info("No cases yet.")
        return

    for case in all_cases:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([2, 2, 2, 3])
            c1.write(f"**{case['case_id']}**")
            c1.caption(case["transaction_id"])
            c2.write(f"Priority: **{case['priority']}**")
            c2.caption(f"Created {case['created_at'][:19]}")
            new_status = c3.selectbox("Status", STATUSES, index=STATUSES.index(case["status"]), key=f"status_{case['case_id']}")
            new_notes = c4.text_area("Notes", value=case["notes"] or "", key=f"notes_{case['case_id']}", height=80)

            if st.button("Save", key=f"save_{case['case_id']}"):
                case_store.update_case(case["case_id"], status=new_status, notes=new_notes)
                st.rerun()

            related = df[df["transaction_id"] == case["transaction_id"]]
            if len(related):
                row = related.iloc[0]
                with st.expander("Related transaction, entities, graph & explanation"):
                    st.write(f"Amount: ${row['amount']:.2f} · Risk: {row['risk_level']} ({row['gnn_score_pct']:.1f}%) · Card: {row['card_key']} · Merchant: {row['merchant_id']}")
                    if st.button("Open full investigation →", key=f"inv_{case['case_id']}"):
                        st.session_state["investigation_txn_id"] = case["transaction_id"]
                        st.session_state["nav_page"] = "Investigation"
                        st.rerun()
