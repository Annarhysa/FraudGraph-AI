"""
Convert the raw IBM TabFormer transactions CSV into Parquet, then carve out
a manageable working sample for graph construction and modeling.

The raw CSV is ~24.4M rows / 2.3GB, which is too slow to iterate on directly.
DuckDB reads/transforms it out-of-core so this step doesn't need the whole
file in memory.

Usage:
    python -m fraudgraph.data_prep
"""
from pathlib import Path

import duckdb

RAW_CSV = Path("data/data_transaction.csv")
FULL_PARQUET = Path("data/processed/transactions_full.parquet")
SAMPLE_PARQUET = Path("data/processed/transactions_sample.parquet")

SAMPLE_USERS = 300  # number of distinct users to keep in the working sample
SAMPLE_START_DATE = "2019-01-01"  # only keep recent transactions so the sample stays small
SAMPLE_SEED = 42  # fixes the filler-user sample so a clean rebuild reproduces the exact dataset


def convert_to_parquet(con: duckdb.DuckDBPyConnection) -> None:
    if FULL_PARQUET.exists():
        print(f"[skip] {FULL_PARQUET} already exists")
        return

    FULL_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    print(f"Reading {RAW_CSV} and writing {FULL_PARQUET} ...")
    con.execute(
        f"""
        COPY (
            SELECT
                "User"            AS user_id,
                "Card"            AS card_id,
                make_date("Year", "Month", "Day")            AS txn_date,
                "Time"            AS txn_time,
                CAST(replace("Amount", '$', '') AS DOUBLE)   AS amount,
                "Use Chip"        AS channel,
                "Merchant Name"   AS merchant_id,
                "Merchant City"   AS merchant_city,
                "Merchant State"  AS merchant_state,
                "Zip"             AS zip,
                "MCC"             AS mcc,
                "Errors?"         AS errors,
                ("Is Fraud?" = 'Yes')                        AS is_fraud
            FROM read_csv_auto('{RAW_CSV.as_posix()}', header=True)
        ) TO '{FULL_PARQUET.as_posix()}' (FORMAT PARQUET);
        """
    )
    print("Done.")


def print_eda(con: duckdb.DuckDBPyConnection) -> None:
    print("\n--- EDA on full dataset ---")
    stats = con.execute(
        f"""
        SELECT
            count(*)                          AS n_txns,
            count(DISTINCT user_id)           AS n_users,
            count(DISTINCT card_id)           AS n_cards,   -- not globally unique, see note below
            count(DISTINCT merchant_id)       AS n_merchants,
            min(txn_date)                     AS min_date,
            max(txn_date)                     AS max_date,
            sum(CASE WHEN is_fraud THEN 1 ELSE 0 END) AS n_fraud,
            round(100.0 * sum(CASE WHEN is_fraud THEN 1 ELSE 0 END) / count(*), 4) AS fraud_pct
        FROM '{FULL_PARQUET.as_posix()}'
        """
    ).fetchone()
    labels = [
        "n_txns", "n_users", "n_cards(raw)", "n_merchants",
        "min_date", "max_date", "n_fraud", "fraud_pct",
    ]
    for label, val in zip(labels, stats):
        print(f"  {label:15s}: {val}")


def build_sample(con: duckdb.DuckDBPyConnection) -> None:
    if SAMPLE_PARQUET.exists():
        print(f"[skip] {SAMPLE_PARQUET} already exists")
        return

    print(f"\nBuilding a {SAMPLE_USERS}-user, post-{SAMPLE_START_DATE} working sample ...")
    con.execute(
        f"""
        COPY (
            WITH recent AS (
                SELECT * FROM '{FULL_PARQUET.as_posix()}'
                WHERE txn_date >= DATE '{SAMPLE_START_DATE}'
            ),
            fraud_users AS (
                SELECT DISTINCT user_id FROM recent WHERE is_fraud
            ),
            filler_users AS (
                SELECT DISTINCT user_id
                FROM recent
                WHERE user_id NOT IN (SELECT user_id FROM fraud_users)
                USING SAMPLE reservoir({SAMPLE_USERS} ROWS) REPEATABLE ({SAMPLE_SEED})
            ),
            sample_users AS (
                SELECT user_id FROM fraud_users
                UNION
                SELECT user_id FROM filler_users
            )
            SELECT t.*
            FROM recent t
            JOIN sample_users s USING (user_id)
        ) TO '{SAMPLE_PARQUET.as_posix()}' (FORMAT PARQUET);
        """
    )
    n = con.execute(f"SELECT count(*) FROM '{SAMPLE_PARQUET.as_posix()}'").fetchone()[0]
    print(f"Sample written: {n} transactions -> {SAMPLE_PARQUET}")


def main() -> None:
    con = duckdb.connect()
    # DuckDB's reservoir sampling is only reproducible under REPEATABLE with single-threaded
    # execution -- parallel scan order otherwise still varies the sample run to run.
    con.execute("SET threads = 1")
    convert_to_parquet(con)
    print_eda(con)
    build_sample(con)


if __name__ == "__main__":
    main()
