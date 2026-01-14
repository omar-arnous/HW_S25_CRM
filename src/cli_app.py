# src/cli_app.py

import numpy as np
from .recommender import AccountInfo, Recommender

def pretty_print(title: str, items):
    print("\n" + "=" * 80)
    print(title)
    print("-" * 80)
    if not items:
        print("No data.")
        return
    for x in items:
        if isinstance(x, tuple) and len(x) == 2:
            print(f"- {x[0]}  |  {x[1]}")
        else:
            print(f"- {x}")

def run_cli(recommender: Recommender):
    print("\nEnter account info:")
    country = input("Country: ").strip()
    solution = input("Solution: ").strip()
    stage = input("Opportunity Stage: ").strip()
    is_lead = input("Is Lead (1/0 or yes/no): ").strip()

    acc = AccountInfo(Country=country, solution=solution, opportunity_stage=stage, is_lead=is_lead)

    pretty_print("Top 4 actions by country (historical)", recommender.top4_by_country(acc.Country))
    pretty_print("Top 4 actions by solution (historical)", recommender.top4_by_solution(acc.solution))
    pretty_print("Top 4 actions by country & solution (historical)", recommender.top4_by_country_solution(acc.Country, acc.solution))
    pretty_print("Top 4 actions by DT (probability)", recommender.predict_top4_dt(acc))

    while True:
        ans = input("\nAdd an action to recalculate Top-4 with new weights? (y/n): ").strip().lower()
        if ans != "y":
            break
        last_action = input("Last action type (e.g., email/call/...): ").strip().lower()
        w = input("Last touch weight (0..1), default 0.3: ").strip()
        last_w = float(w) if w else 0.3
        pretty_print("Top 4 actions after adding action (Dynamic Weights)", recommender.recalc_after_action(acc, last_action, last_w))

    while True:
        ans2 = input("\nBuild best trip? (win/loss/skip): ").strip().lower()
        if ans2 in ["skip", "s", "n", "no", ""]:
            break
        if ans2 not in ["win", "loss"]:
            print("Please enter 'win' or 'loss' or 'skip'")
            continue

        horizon = input("Horizon steps (default 5): ").strip()
        horizon = int(horizon) if horizon else 5

        trip = recommender.best_trip(acc, objective=ans2, horizon=horizon)

        print("\n" + "=" * 80)
        print(f"Best Trip ({ans2.upper()}) - Greedy Plan")
        print("-" * 80)
        if not trip:
            print("No trip generated.")
        else:
            for step in trip:
                used = "OutcomeModel" if step["used_outcome_model"] else "FallbackHeuristic"
                pwin_txt = f"{step['p_win']:.3f}" if np.isfinite(step["p_win"]) else "NA"
                print(
                    f"Step {step['step']}: Stage='{step['current_stage']}' "
                    f"-> Action='{step['action']}' "
                    f"-> NextStage='{step['predicted_next_stage']}' "
                    f"| p_win={pwin_txt} | score={step['score']:.3f} | {used}"
                )
