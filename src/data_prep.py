# src/data_prep.py

import numpy as np
import pandas as pd
from .config import WON_KEYS, LOST_KEYS

def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().drop_duplicates()

    critical = ["account_id", "activity_date", "Country", "solution", "opportunity_stage"]
    df = df.dropna(subset=critical)

    df["activity_date"] = pd.to_datetime(df["activity_date"], errors="coerce")
    df = df.dropna(subset=["activity_date"])

    df["Country"] = df["Country"].astype(str).str.strip().str.title()
    df["solution"] = df["solution"].astype(str).str.strip().str.lower()
    df["opportunity_stage"] = df["opportunity_stage"].astype(str).str.strip()
    df["types"] = df["types"].astype(str).str.strip().str.lower()

    if "is_lead" in df.columns:
        df["is_lead"] = df["is_lead"].apply(lambda v: _safe_str(v).lower())
        df["is_lead"] = df["is_lead"].replace(
            {"true": "1", "false": "0", "yes": "1", "no": "0", "y": "1", "n": "0"}
        )

    if "opportunity_id" not in df.columns:
        df["opportunity_id"] = np.nan

    return df

def build_event_key(df: pd.DataFrame) -> pd.Series:
    opp = df["opportunity_id"].astype(str)
    acc = df["account_id"].astype(str)
    opp_missing = opp.isna() | (opp.str.lower().isin(["nan", "none", ""]))
    key = np.where(opp_missing, "ACC_" + acc, "OPP_" + opp)
    return pd.Series(key, index=df.index, name="event_key")

def add_next_stage(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_key"] = build_event_key(df)
    df = df.sort_values(["event_key", "activity_date"])
    df["next_stage"] = df.groupby("event_key")["opportunity_stage"].shift(-1)
    return df

def derive_outcome_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["event_key"] = build_event_key(df)

    def has_any(keys, s: str) -> bool:
        return any(k in s for k in keys)

    outcome_per_key = df.groupby("event_key").apply(
        lambda g: 1
        if g["opportunity_stage"].astype(str).str.lower().apply(lambda x: has_any(WON_KEYS, x)).any()
        else (0
              if g["opportunity_stage"].astype(str).str.lower().apply(lambda x: has_any(LOST_KEYS, x)).any()
              else np.nan)
    )
    outcome_per_key.name = "outcome"
    return df.merge(outcome_per_key.reset_index(), on="event_key", how="left")
