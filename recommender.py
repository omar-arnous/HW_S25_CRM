# src/recommender.py

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from .config import FEATURE_COLS, ACTION_COL, NEXT_STAGE_COL

def _safe_str(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()

def top_k_actions(series: pd.Series, k: int = 4) -> List[Tuple[str, int]]:
    vc = series.value_counts()
    vc = vc[vc.index.notna()]
    return list(zip(vc.head(k).index.tolist(), vc.head(k).values.tolist()))

@dataclass
class AccountInfo:
    Country: str
    solution: str
    opportunity_stage: str
    is_lead: str

class Recommender:
    def __init__(self, df: pd.DataFrame, action_model, stage_model, outcome_model):
        self.df = df
        self.action_model = action_model
        self.stage_model = stage_model
        self.outcome_model = outcome_model

        self.all_actions = sorted(self.df[ACTION_COL].dropna().unique().tolist())

        self._precompute_tables()

    def _precompute_tables(self):
        d = self.df.dropna(subset=[ACTION_COL, "Country", "solution"]).copy()
        self.top_by_country = d.groupby("Country")[ACTION_COL].apply(lambda s: top_k_actions(s, 4)).to_dict()
        self.top_by_solution = d.groupby("solution")[ACTION_COL].apply(lambda s: top_k_actions(s, 4)).to_dict()
        self.top_by_country_solution = d.groupby(["Country", "solution"])[ACTION_COL].apply(lambda s: top_k_actions(s, 4)).to_dict()

    def top4_by_country(self, country: str):
        return self.top_by_country.get(_safe_str(country).title(), [])

    def top4_by_solution(self, solution: str):
        return self.top_by_solution.get(_safe_str(solution).lower(), [])

    def top4_by_country_solution(self, country: str, solution: str):
        return self.top_by_country_solution.get((_safe_str(country).title(), _safe_str(solution).lower()), [])

    def _to_X(self, acc: AccountInfo) -> pd.DataFrame:
        return pd.DataFrame([{
            "Country": _safe_str(acc.Country).title(),
            "solution": _safe_str(acc.solution).lower(),
            "opportunity_stage": _safe_str(acc.opportunity_stage),
            "is_lead": _safe_str(acc.is_lead).lower(),
        }])

    def predict_top4_dt(self, acc: AccountInfo) -> List[Tuple[str, float]]:
        if self.action_model is None:
            return []
        X = self._to_X(acc)
        proba = self.action_model.predict_proba(X)[0]
        classes = self.action_model.named_steps["dt"].classes_
        pairs = sorted(list(zip(classes, proba)), key=lambda x: x[1], reverse=True)
        return [(a, float(p)) for a, p in pairs[:4]]

    def predict_all_probs(self, acc: AccountInfo) -> Dict[str, float]:
        if self.action_model is None:
            return {}
        X = self._to_X(acc)
        proba = self.action_model.predict_proba(X)[0]
        classes = self.action_model.named_steps["dt"].classes_
        return {str(c): float(p) for c, p in zip(classes, proba)}

    def recalc_after_action(self, acc: AccountInfo, last_action: str, last_touch_weight: float = 0.3):
        base = self.predict_all_probs(acc)
        la = _safe_str(last_action).lower()
        if la in base:
            base[la] = base[la] * (1 - float(last_touch_weight))
        total = sum(base.values())
        if total > 0:
            base = {k: v / total for k, v in base.items()}
        return sorted(base.items(), key=lambda x: x[1], reverse=True)[:4]

    def best_trip(self, acc: AccountInfo, objective: str = "win", horizon: int = 5):
        if self.stage_model is None:
            return []

        use_outcome = self.outcome_model is not None
        objective = objective.lower().strip()

        country = _safe_str(acc.Country).title()
        solution = _safe_str(acc.solution).lower()
        is_lead = _safe_str(acc.is_lead).lower()
        stage = _safe_str(acc.opportunity_stage)

        trip = []
        terminal = ["closed won", "closed lost", "won", "lost"]

        for step in range(1, horizon + 1):
            if any(t in stage.lower() for t in terminal):
                break

            best = None
            for action in self.all_actions:
                # predict next stage
                X_stage = pd.DataFrame([{
                    "Country": country,
                    "solution": solution,
                    "opportunity_stage": stage,
                    "is_lead": is_lead,
                    "types": action
                }])
                next_stage = self.stage_model.predict(X_stage)[0]

                if use_outcome:
                    X_out = X_stage.copy()
                    proba = self.outcome_model.predict_proba(X_out)[0]
                    classes = self.outcome_model.named_steps["dt"].classes_
                    p_win = float(proba[list(classes).index(1)]) if 1 in classes else 0.0
                    score = p_win if objective == "win" else (1.0 - p_win)
                else:
                    ns = str(next_stage).lower()
                    won_like = any(k in ns for k in ["won", "win", "successful", "success"])
                    lost_like = any(k in ns for k in ["lost", "loss", "unsuccessful", "failed", "failure"])
                    if objective == "win":
                        score = 1.0 if won_like else (0.0 if lost_like else 0.5)
                    else:
                        score = 1.0 if lost_like else (0.0 if won_like else 0.5)
                    p_win = float("nan")

                cand = {
                    "step": step,
                    "current_stage": stage,
                    "action": action,
                    "predicted_next_stage": str(next_stage),
                    "score": float(score),
                    "p_win": float(p_win),
                    "used_outcome_model": bool(use_outcome)
                }
                if best is None or cand["score"] > best["score"]:
                    best = cand

            if best is None:
                break

            trip.append(best)
            stage = best["predicted_next_stage"]

        return trip
