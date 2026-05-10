import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class TrainingRun:
    name: str
    year: float
    flops: float
    parameters_b: float


HISTORICAL_RUNS = [
    TrainingRun("GPT-2", 2019.1, 1.5e21, 1.5),
    TrainingRun("GPT-3", 2020.4, 3.14e23, 175.0),
    TrainingRun("Gopher", 2021.9, 5.76e23, 280.0),
    TrainingRun("PaLM", 2022.3, 2.56e24, 540.0),
    TrainingRun("GPT-4 (est.)", 2023.2, 2.15e25, 1000.0),
    TrainingRun("Gemini Ultra (est.)", 2023.8, 5.0e25, 1800.0),
    TrainingRun("Frontier 2024 (est.)", 2024.5, 1.0e26, 3000.0),
    TrainingRun("Frontier 2025 (est.)", 2025.3, 5.0e26, 8000.0),
]


class ComputeDemandModel:
    H100_FLOPS_PER_SEC = 3.958e15   # FP16 FLOPS/s
    H100_WATTS = 700                 # TDP watts
    H100_PRICE_USD = 30_000
    PUE = 1.4                        # typical data center power overhead

    def training_power_mw(self, total_flops: float, training_days: float = 90) -> float:
        """Total facility power in MW for a single training run."""
        seconds = training_days * 86_400
        chips = total_flops / (self.H100_FLOPS_PER_SEC * seconds)
        return chips * self.H100_WATTS * self.PUE / 1e6

    def chips_needed(self, total_flops: float, training_days: float = 90) -> int:
        seconds = training_days * 86_400
        return int(total_flops / (self.H100_FLOPS_PER_SEC * seconds))

    def historical_compute_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"name": r.name, "year": r.year, "flops": r.flops, "parameters_b": r.parameters_b}
            for r in HISTORICAL_RUNS
        ])

    def project_industry_power_gw(self, years: int = 8) -> pd.DataFrame:
        """
        Project total AI industry power demand in GW.
        Assumes ~5 frontier training clusters in 2024 growing ~80%/yr,
        with inference load scaling alongside.
        """
        base_year = 2024
        base_clusters = 5
        cluster_growth = 1.80
        base_flops = 2e25
        # frontier compute doubles roughly every 6 months → ×4 per year
        compute_growth = 4.0

        rows = []
        for y in range(years + 1):
            year = base_year + y
            n_clusters = base_clusters * (cluster_growth ** y)
            flops_per_cluster = base_flops * (compute_growth ** y)
            training_gw = self.training_power_mw(flops_per_cluster) * n_clusters / 1000
            # inference grows relative to training; rough multiplier
            inference_mult = 1.0 + y * 0.45
            total_gw = training_gw * inference_mult
            rows.append({
                "year": year,
                "training_power_gw": training_gw,
                "total_power_gw": total_gw,
                "n_clusters": n_clusters,
                "flops_per_cluster": flops_per_cluster,
            })
        return pd.DataFrame(rows)

    def power_table(self) -> pd.DataFrame:
        scenarios = [
            ("GPT-3 (2020)", 3.14e23),
            ("GPT-4 (2023)", 2.15e25),
            ("Frontier 2024 (est.)", 1.0e26),
            ("Frontier 2025 (est.)", 5.0e26),
            ("Frontier 2026 (est.)", 2.0e27),
            ("Frontier 2028 (est.)", 1.0e28),
        ]
        rows = []
        for name, flops in scenarios:
            mw = self.training_power_mw(flops, 90)
            chips = self.chips_needed(flops, 90)
            rows.append({
                "Model": name,
                "Training FLOPs": f"{flops:.1e}",
                "H100-equiv Chips": f"{chips:,}",
                "Training Power (MW)": f"{mw:.0f}",
                "Equivalent Scale": f"{mw/1000:.1f} GW" if mw >= 1000 else f"{mw:.0f} MW",
            })
        return pd.DataFrame(rows)
