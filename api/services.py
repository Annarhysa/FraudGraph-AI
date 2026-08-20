"""
Plain-Python (non-Streamlit) cached data/model access for the FastAPI
layer — mirrors src/fraudgraph/store.py but uses functools.lru_cache
instead of st.cache_*, since this runs outside a Streamlit runtime.
"""
import sys
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from fraudgraph import cases as case_store
from fraudgraph import clusters as cluster_engine
from fraudgraph import gnn
from fraudgraph import scoring


@lru_cache(maxsize=1)
def get_alerts_df() -> pd.DataFrame:
    return scoring.build_master()


@lru_cache(maxsize=1)
def get_graph_and_model():
    df = get_alerts_df()
    data, meta = gnn.build_graph(df)
    train_idx, val_idx, test_idx = gnn.temporal_split(meta["n_txn"], meta["txn_offset"])
    model = gnn.load_or_train_model(data, meta, train_idx, val_idx)
    return data, meta, model


@lru_cache(maxsize=1)
def get_clusters():
    df = get_alerts_df()
    return cluster_engine.detect_clusters(df)


def merged_alerts() -> pd.DataFrame:
    df = get_alerts_df().copy()
    statuses = case_store.get_alert_statuses()
    if statuses:
        df["status"] = df["transaction_id"].map(statuses).fillna(df["status"])
    return df


def row_to_json(row: pd.Series) -> dict:
    out = row.to_dict()
    for k, v in out.items():
        if isinstance(v, pd.Timestamp):
            out[k] = v.isoformat()
        elif hasattr(v, "item"):  # numpy scalar
            v = v.item()
            out[k] = None if isinstance(v, float) and pd.isna(v) else v
        elif isinstance(v, float) and pd.isna(v):
            out[k] = None
    return out


def df_to_json(df: pd.DataFrame) -> list[dict]:
    return [row_to_json(r) for _, r in df.iterrows()]
