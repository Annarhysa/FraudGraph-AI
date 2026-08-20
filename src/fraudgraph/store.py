"""
Central cached data/model access for the Streamlit app. Every page imports
from here instead of touching parquet files, the graph, or the model
directly — keeps the app from rebuilding the graph or retraining on every
click, and keeps every page's numbers consistent.
"""
import streamlit as st

from . import batches as batch_store
from . import cases as case_store
from . import clusters as cluster_engine
from . import explainability
from . import gnn
from . import scoring


@st.cache_data(show_spinner="Loading alerts...")
def load_alerts():
    return scoring.build_master()


@st.cache_resource(show_spinner="Loading graph + model...")
def load_graph_and_model():
    df = load_alerts()
    data, meta = gnn.build_graph(df)
    train_idx, val_idx, test_idx = gnn.temporal_split(meta["n_txn"], meta["txn_offset"])
    model = gnn.load_or_train_model(data, meta, train_idx, val_idx)
    return data, meta, model, train_idx, val_idx, test_idx


@st.cache_data(show_spinner="Detecting suspicious clusters...")
def load_clusters():
    df = load_alerts()
    clusters_df, members_df = cluster_engine.detect_clusters(df)
    return clusters_df, members_df


@st.cache_resource
def get_model_explainer():
    data, meta, model, *_ = load_graph_and_model()
    return explainability.ModelExplainer(data, meta, model)


def get_alert_statuses() -> dict:
    return case_store.get_alert_statuses()


def merged_alerts():
    """Alerts table with any analyst decisions (OPEN/CONFIRMED/DISMISSED/...) applied."""
    df = load_alerts().copy()
    statuses = get_alert_statuses()
    if statuses:
        df["status"] = df["transaction_id"].map(statuses).fillna(df["status"])
    return df


# ---------------------------------------------------------------------------
# Active batch (the user's uploaded data) — every dashboard page reads from
# here, not from the shared sample above. The shared sample/model/calibration
# still exist underneath (that's what score_uploaded() scores a batch with),
# but "what the dashboard shows" is per-user, per-upload.
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_batch_df(batch_id: str):
    df = batch_store.load_batch(batch_id)
    statuses = get_alert_statuses()
    if statuses:
        df["status"] = df["transaction_id"].map(statuses).fillna(df["status"])
    return df


@st.cache_resource(show_spinner=False)
def _build_batch_graph(batch_id: str):
    df = batch_store.load_batch(batch_id)
    return gnn.build_graph(df)


def active_batch_id():
    return st.session_state.get("active_batch_id")


def invalidate_active_df():
    """Call after writing an alert-status change so active_df() picks it up."""
    _load_batch_df.clear()


def active_df():
    batch_id = active_batch_id()
    if batch_id is None:
        return None
    return _load_batch_df(batch_id)


def active_graph():
    batch_id = active_batch_id()
    if batch_id is None:
        return None, None
    return _build_batch_graph(batch_id)


@st.cache_resource(show_spinner=False)
def _active_model_explainer(batch_id: str):
    data, meta = _build_batch_graph(batch_id)
    _, _, model, *_ = load_graph_and_model()  # shared pretrained model
    return explainability.ModelExplainer(data, meta, model)


def active_model_explainer():
    batch_id = active_batch_id()
    if batch_id is None:
        return None
    return _active_model_explainer(batch_id)


def active_clusters():
    df = active_df()
    if df is None or df.empty:
        return None, None
    return cluster_engine.compute_clusters(df)
