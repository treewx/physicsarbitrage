import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class EnergyBottleneck:
    name: str
    description: str
    lead_time_years: float
    current_capacity_gw: float
    annual_capacity_add_gw: float
    constraint_severity: float       # 1–10
    investable_sectors: List[str] = field(default_factory=list)


BOTTLENECKS = [
    EnergyBottleneck(
        name="Grid Interconnection Queue",
        description=(
            "US interconnection queue backlog exceeds 2,600 GW of proposed projects. "
            "Average wait: 5–7 years. Pre-connected sites command massive premiums."
        ),
        lead_time_years=6.0,
        current_capacity_gw=50.0,
        annual_capacity_add_gw=5.0,
        constraint_severity=9.5,
        investable_sectors=[
            "Electrical contractors (PWR, MYR)",
            "Transmission builders",
            "Utilities with pre-permitted capacity (VST, CEG)",
            "Crypto miners with connected sites (CORZ, WULF, IREN)",
        ],
    ),
    EnergyBottleneck(
        name="Power Transformer Manufacturing",
        description=(
            "Large power transformers (LPTs) have 1–2 year delivery times. "
            "The US has no domestic LPT manufacturer; imports from Germany, South Korea, Mexico."
        ),
        lead_time_years=2.0,
        current_capacity_gw=30.0,
        annual_capacity_add_gw=3.0,
        constraint_severity=8.5,
        investable_sectors=[
            "Electrical equipment manufacturers (ETN, HUBB)",
            "Grid technology companies (GEV)",
            "Transformer importers",
        ],
    ),
    EnergyBottleneck(
        name="New Generation Capacity",
        description=(
            "US needs 400–800 GW of new generation by 2030 to support AI + electrification. "
            "Current build rate is ~60 GW/year. Permits, gas supply, and construction all bottleneck."
        ),
        lead_time_years=4.0,
        current_capacity_gw=60.0,
        annual_capacity_add_gw=15.0,
        constraint_severity=9.0,
        investable_sectors=[
            "Generators with existing capacity (VST, NRG, CEG)",
            "Gas turbine manufacturers (GEV)",
            "Nuclear operators",
            "Fuel cell providers (BE)",
        ],
    ),
    EnergyBottleneck(
        name="Skilled Electrical Labor",
        description=(
            "300,000+ additional electricians needed by 2030. "
            "Apprenticeship programs take 4–5 years. Structural shortage compounds grid delays."
        ),
        lead_time_years=4.5,
        current_capacity_gw=0.0,
        annual_capacity_add_gw=0.0,
        constraint_severity=7.0,
        investable_sectors=[
            "Electrical contractors (PWR, MYRG)",
            "Industrial automation companies",
            "Workforce training platforms",
        ],
    ),
    EnergyBottleneck(
        name="Data Center Liquid Cooling",
        description=(
            "AI chips dissipate 700W–1,200W each. Air cooling maxes out at ~50 kW/rack. "
            "Liquid cooling is physically mandatory for GB200 and next-gen AI clusters."
        ),
        lead_time_years=1.5,
        current_capacity_gw=5.0,
        annual_capacity_add_gw=8.0,
        constraint_severity=8.0,
        investable_sectors=[
            "Liquid cooling manufacturers (VRT)",
            "Thermal management companies (MOD)",
            "Coolant fluid suppliers",
            "Server OEMs with liquid-cooling lines (SMCI)",
        ],
    ),
    EnergyBottleneck(
        name="Nuclear Baseload",
        description=(
            "AI data centers need 24/7 firm power. Nuclear is the only scalable carbon-free baseload. "
            "US fleet is aging; SMRs are 5–10 years away from commercial deployment."
        ),
        lead_time_years=8.0,
        current_capacity_gw=100.0,
        annual_capacity_add_gw=2.0,
        constraint_severity=7.5,
        investable_sectors=[
            "Uranium miners (CCJ, UEC)",
            "Nuclear operators (CEG, VST)",
            "SMR developers (OKLO, NuScale SMR)",
            "Uranium enrichers (LEU)",
        ],
    ),
]


class EnergyBottleneckModel:
    def __init__(self):
        self.bottlenecks = BOTTLENECKS

    def demand_supply_gap(self, base_ai_demand_gw: float = 15.0, years: int = 8) -> pd.DataFrame:
        rows = []
        for y in range(years + 1):
            year = 2024 + y
            demand = base_ai_demand_gw * (1.70 ** y)
            supply = 15.0 + y * 30.0
            rows.append({
                "year": year,
                "demand_gw": demand,
                "supply_gw": supply,
                "gap_gw": max(0.0, demand - supply),
            })
        return pd.DataFrame(rows)

    def bottleneck_radar_df(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "bottleneck": b.name,
                "severity": b.constraint_severity,
                "investability": round(10.0 - b.lead_time_years, 1),
                "lead_time": b.lead_time_years,
            }
            for b in self.bottlenecks
        ])
