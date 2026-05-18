import os
import pandas as pd

# ── Scoring formula weights ──────────────────────────────────────────────────
# Physics Score = max(ownership_score, supplier_score) * OWNERSHIP_WT
#                 + supply_response_score * SUPPLY_WT
#
# ownership_score  — does the company OWN the scarce physical asset?
# supplier_score   — is the company the MANDATORY SUPPLIER of a critical component?
# The formula rewards whichever strength is higher; most companies score well on
# one axis and low on the other.  Both inputs are on a 1-10 scale; weights sum to 1.0.
OWNERSHIP_WT = 0.55
SUPPLY_WT = 0.45

# Path to the editable company database
_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "companies.csv")


def load_companies() -> pd.DataFrame:
    """
    Load company database from data/companies.csv and compute physics_score.

    Physics Score = max(ownership_score, supplier_score) * OWNERSHIP_WT
                    + supply_response_score * SUPPLY_WT

    For companies that pre-date the supplier_score column, supplier_score is
    treated as absent and the formula falls back to ownership_score alone,
    so existing scores are unchanged until a company is re-evaluated.
    """
    df = pd.read_csv(_CSV_PATH)

    # Back-fill columns added after initial CSV creation
    if "supplier_score" not in df.columns:
        df["supplier_score"] = float("nan")
    if "strategy_type" not in df.columns:
        df["strategy_type"] = ""

    # Effective strength = max(ownership, supplier) where supplier has been set
    _eff = df["ownership_score"].where(
        df["supplier_score"].isna(),
        df[["ownership_score", "supplier_score"]].max(axis=1),
    )
    df["Physics Score"] = (_eff * OWNERSHIP_WT + df["supply_response_score"] * SUPPLY_WT).round(1)

    # Rename columns to display-friendly names
    df = df.rename(columns={
        "ticker":                  "Ticker",
        "name":                    "Name",
        "category":                "Category",
        "subcategory":             "Subcategory",
        "strategy_type":           "Strategy Type",
        "cycle_stage":             "Cycle Stage",
        "lead_time_advantage_yrs": "Lead Time Adv (yrs)",
        "ownership_score":         "Ownership Score",
        "supplier_score":          "Supplier Score",
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
