import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Technology:
    name: str
    initial_cost: float
    initial_volume: float
    learning_rate: float
    unit: str
    current_volume: Optional[float] = None
    current_cost: Optional[float] = None
    annual_growth_rate: float = 0.20
    description: str = ""
    color: str = "#4CAF50"


TECHNOLOGY_DATABASE = [
    Technology(
        name="Solar PV",
        initial_cost=100.0,
        initial_volume=0.1,
        learning_rate=0.22,
        unit="$/W",
        current_volume=1200.0,
        current_cost=0.22,
        annual_growth_rate=0.35,
        description="Solar panel cost per watt — following Wright's Law since 1976",
        color="#FFA726",
    ),
    Technology(
        name="Lithium-Ion Batteries",
        initial_cost=1200.0,
        initial_volume=0.5,
        learning_rate=0.18,
        unit="$/kWh",
        current_volume=2500.0,
        current_cost=120.0,
        annual_growth_rate=0.40,
        description="Battery pack cost per kWh — EV and grid storage learning curve",
        color="#42A5F5",
    ),
    Technology(
        name="AI Chips ($/TFLOP/s)",
        initial_cost=5000.0,
        initial_volume=0.01,
        learning_rate=0.40,
        unit="$/TFLOP/s",
        current_volume=100.0,
        current_cost=3.0,
        annual_growth_rate=0.50,
        description="GPU compute cost per TFLOP/s — accelerating faster than Moore's Law",
        color="#AB47BC",
    ),
    Technology(
        name="Onshore Wind",
        initial_cost=4.5,
        initial_volume=1.0,
        learning_rate=0.07,
        unit="$/W",
        current_volume=950.0,
        current_cost=0.90,
        annual_growth_rate=0.10,
        description="Wind turbine capex per watt — slower learning curve than solar",
        color="#66BB6A",
    ),
    Technology(
        name="Electrolyzers (Green H2)",
        initial_cost=2000.0,
        initial_volume=0.01,
        learning_rate=0.15,
        unit="$/kW",
        current_volume=5.0,
        current_cost=800.0,
        annual_growth_rate=0.60,
        description="Electrolyzer cost for green hydrogen — very early on the curve",
        color="#26C6DA",
    ),
    Technology(
        name="Nuclear SMR",
        initial_cost=12000.0,
        initial_volume=0.001,
        learning_rate=0.10,
        unit="$/kW",
        current_volume=0.01,
        current_cost=10000.0,
        annual_growth_rate=0.80,
        description="Small Modular Reactor capex — barely started on the learning curve",
        color="#EF5350",
    ),
]


class WrightLawModel:
    """
    Wright's Law: cost declines by a fixed fraction for each doubling of cumulative production.
    Cost(N) = Cost_0 * (N/N_0)^(-alpha), where alpha = -log2(1 - learning_rate)
    """

    def __init__(self, tech: Technology):
        self.tech = tech
        self.alpha = -np.log2(1 - tech.learning_rate)

    def cost_at_volume(self, volume: float) -> float:
        return self.tech.initial_cost * (volume / self.tech.initial_volume) ** (-self.alpha)

    def volume_for_cost(self, target_cost: float) -> float:
        ratio = target_cost / self.tech.initial_cost
        return self.tech.initial_volume * (ratio ** (-1.0 / self.alpha))

    def years_to_cost(self, target_cost: float, current_volume: float, annual_growth: float) -> float:
        target_volume = self.volume_for_cost(target_cost)
        if target_volume <= current_volume:
            return 0.0
        return np.log(target_volume / current_volume) / np.log(1 + annual_growth)

    def project_forward(self, years: int, current_volume: float, annual_growth: float) -> pd.DataFrame:
        year_range = np.arange(0, years + 1)
        volumes = current_volume * (1 + annual_growth) ** year_range
        costs = [self.cost_at_volume(v) for v in volumes]
        base_cost = costs[0]
        return pd.DataFrame({
            "year": 2024 + year_range,
            "volume": volumes,
            "cost": costs,
            "cost_reduction_pct": [(c / base_cost - 1) * 100 for c in costs],
        })

    def historical_curve(self) -> pd.DataFrame:
        max_vol = max(
            self.tech.current_volume or self.tech.initial_volume * 1000,
            self.tech.initial_volume * 10,
        )
        volumes = np.logspace(
            np.log10(self.tech.initial_volume),
            np.log10(max_vol),
            120,
        )
        costs = [self.cost_at_volume(v) for v in volumes]
        return pd.DataFrame({"volume": volumes, "cost": costs})
