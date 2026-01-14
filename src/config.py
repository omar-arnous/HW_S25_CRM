# src/config.py

FEATURE_COLS = ["Country", "solution", "opportunity_stage", "is_lead"]
ACTION_COL = "types"
NEXT_STAGE_COL = "next_stage"

DEFAULT_FILENAME = "data_all.xltx"

# Flexible outcome keywords
WON_KEYS = ["closed won", "won", "win", "successful", "success"]
LOST_KEYS = ["closed lost", "lost", "loss", "unsuccessful", "failed", "failure"]

# Decision Tree hyperparameters
DT_ACTION_PARAMS = dict(max_depth=6, min_samples_leaf=30, random_state=42)
DT_STAGE_PARAMS  = dict(max_depth=6, min_samples_leaf=30, random_state=42)
DT_OUTCOME_PARAMS = dict(max_depth=6, min_samples_leaf=30, random_state=42)

DEFAULT_TEST_SIZE = 0.2
DEFAULT_LAST_TOUCH_WEIGHT = 0.3
DEFAULT_TRIP_HORIZON = 5
