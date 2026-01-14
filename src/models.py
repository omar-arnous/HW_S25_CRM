# src/models.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from .config import (
    DT_ACTION_PARAMS, DT_STAGE_PARAMS, DT_OUTCOME_PARAMS,
    DEFAULT_TEST_SIZE
)

def make_dt_pipeline(feature_cols: List[str], dt_params: dict) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[("cat", OneHotEncoder(handle_unknown="ignore"), feature_cols)],
        remainder="drop"
    )
    dt = DecisionTreeClassifier(**dt_params)
    return Pipeline(steps=[("pre", pre), ("dt", dt)])

def train_classifier(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    dt_params: dict,
    test_size: float = DEFAULT_TEST_SIZE,
    stratify: bool = True
) -> Dict[str, object]:
    dfm = df.dropna(subset=feature_cols + [target_col]).copy()
    X = dfm[feature_cols]
    y = dfm[target_col]

    if dfm.empty:
        return {"model": None, "accuracy": float("nan"), "note": "No training rows after dropping NaNs."}

    do_stratify = stratify and (y.nunique() > 1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42,
        stratify=y if do_stratify else None
    )

    model = make_dt_pipeline(feature_cols, dt_params)
    model.fit(X_train, y_train)

    preds = model.predict(X_test) if len(X_test) else []
    acc = accuracy_score(y_test, preds) if len(y_test) else float("nan")
    return {"model": model, "accuracy": float(acc)}

def train_action_model(df: pd.DataFrame, feature_cols: List[str], action_col: str) -> Dict[str, object]:
    return train_classifier(df, feature_cols, action_col, DT_ACTION_PARAMS)

def train_stage_model(df: pd.DataFrame, feature_cols: List[str], next_stage_col: str) -> Dict[str, object]:
    return train_classifier(df, feature_cols, next_stage_col, DT_STAGE_PARAMS)

def train_outcome_model(df: pd.DataFrame, feature_cols: List[str], outcome_col: str) -> Dict[str, object]:
    dfm = df.dropna(subset=feature_cols + [outcome_col]).copy()
    dfm = dfm[dfm[outcome_col].isin([0, 1])]
    counts = dfm[outcome_col].value_counts().to_dict()

    if dfm.empty or len(counts) < 2:
        return {
            "model": None,
            "accuracy": float("nan"),
            "counts": counts,
            "note": "Not enough WON/LOST labels to train outcome model."
        }

    out = train_classifier(dfm, feature_cols, outcome_col, DT_OUTCOME_PARAMS)
    out["counts"] = counts
    return out
