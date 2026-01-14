# main.py

import pandas as pd

from src.config import DEFAULT_FILENAME, FEATURE_COLS, ACTION_COL, NEXT_STAGE_COL
from src.data_prep import clean_data, add_next_stage, derive_outcome_labels
from src.models import train_action_model, train_stage_model, train_outcome_model
from src.recommender import Recommender
from src.cli_app import run_cli

def main():
    filename = DEFAULT_FILENAME
    df = pd.read_excel(filename,sheet_name="data")

    # Prepare data
    df = clean_data(df)
    df = add_next_stage(df)
    df = derive_outcome_labels(df)

    print("\nTop opportunity_stage values (lowercased):")
    print(df["opportunity_stage"].astype(str).str.lower().value_counts().head(20))
    print("\nOutcome distribution (including NaN):")
    print(df["outcome"].value_counts(dropna=False))

    # Train models
    print("\nTraining models ...")

    action_out = train_action_model(df, FEATURE_COLS, ACTION_COL)
    action_model = action_out["model"]
    print({"action_model_accuracy": action_out["accuracy"]})

    stage_features = FEATURE_COLS + [ACTION_COL]
    stage_out = train_stage_model(df.dropna(subset=[NEXT_STAGE_COL]), stage_features, NEXT_STAGE_COL)
    stage_model = stage_out["model"]
    print({"stage_transition_model_accuracy": stage_out["accuracy"]})

    outcome_out = train_outcome_model(df, stage_features, "outcome")
    outcome_model = outcome_out["model"]
    print({
        "outcome_model_accuracy": outcome_out.get("accuracy"),
        "outcome_label_counts": outcome_out.get("counts"),
        "note": outcome_out.get("note")
    })

    # Recommender
    rec = Recommender(df=df, action_model=action_model, stage_model=stage_model, outcome_model=outcome_model)

    # Run CLI
    run_cli(rec)

if __name__ == "__main__":
    main()
