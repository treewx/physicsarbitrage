import os
import pandas as pd

# ── Scoring formula weights ──────────────────────────────────────────────────
# Physics Score = ownership_score * OWNERSHIP_WT + supply_response_score * SUPPLY_WT
# Both inputs are on a 1-10 scale; weights sum to 1.0.
OWNERSHIP_WT = 0.55
SUPPLY_WT = 0.45

# Path to the editable company database
_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "companies.csv")


def load_companies() -> pd.DataFrame:
    """
    Load company database from data/companies.csv and compute physics_score.

    To add, remove, or edit companies — open data/companies.csv in Excel or
    Google Sheets.  The physics_score column is calculated automatically from
    ownership_score and supply_response_score using the formula above.
    """
    df = pd.read_csv(_CSV_PATH)

    # Compute physics score from component scores
    df["Physics Score"] = (
        df["ownership_score"] * OWNERSHIP_WT
        + df["supply_response_score"] * SUPPLY_WT
    ).round(1)

    # Rename columns to display-friendly names
    df = df.rename(columns={
        "ticker":                  "Ticker",
        "name":                    "Name",
        "category":                "Category",
        "subcategory":             "Subcategory",
        "cycle_stage":             "Cycle Stage",
        "lead_time_advantage_yrs": "Lead Time Adv (yrs)",
        "ownership_score":         "Ownership Score",
        "supply_response_score":   "Supply Constraint Score",
        "constraint_controlled":   "Constraint Controlled",
        "physics_thesis":          "Thesis",
        "key_metrics":             "Key Metrics",
    })

    return df


def categories() -> list:
    df = load_companies()
    return sorted(df["Category"].unique().tolist())


class OpportunityScreener:
    def __init__(self):
        self._df = load_companies()

    def as_dataframe(self) -> pd.DataFrame:
        return self._df.copy()

    def categories(self) -> list:
        return sorted(self._df["Category"].unique().tolist())

    def top_by_score(self, n: int = 10) -> pd.DataFrame:
        return self._df.nlargest(n, "Physics Score")
