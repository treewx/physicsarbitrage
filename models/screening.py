import pandas as pd
from dataclasses import dataclass, field
from typing import List


@dataclass
class InfraCompany:
    ticker: str
    name: str
    category: str
    subcategory: str
    physics_thesis: str
    constraint_controlled: str
    physics_score: float        # 1–10
    cycle_stage: str            # "early" | "middle" | "late" | "mature"
    lead_time_advantage: float  # years before competition can replicate
    key_metrics: List[str] = field(default_factory=list)


COMPANY_DATABASE: List[InfraCompany] = [

    # ── POWER GENERATION ──────────────────────────────────────────────────────
    InfraCompany(
        ticker="BE",
        name="Bloom Energy",
        category="Power Generation",
        subcategory="Fuel Cells",
        physics_thesis=(
            "Solid-oxide fuel cells convert natural gas → DC electricity on-site, "
            "bypassing the 5–7 year grid interconnection queue. Aschenbrenner's largest position ($855M). "
            "Backlog exceeds $20B; revenue growing ~34% in 2025, ~40% guided for 2026."
        ),
        constraint_controlled="On-site power generation that bypasses grid connection queues",
        physics_score=9.5,
        cycle_stage="early",
        lead_time_advantage=5.0,
        key_metrics=["Revenue growth", "Backlog size", "Electrolyzer optionality"],
    ),
    InfraCompany(
        ticker="VST",
        name="Vistra Energy",
        category="Power Generation",
        subcategory="Merchant Power",
        physics_thesis=(
            "Largest competitive power generator in the US. Nuclear + gas fleet provides 24/7 firm power "
            "that hyperscalers desperately need. Has direct corporate PPA pipeline with data center tenants."
        ),
        constraint_controlled="Existing permitted power generation capacity — takes decades to replicate",
        physics_score=8.5,
        cycle_stage="early",
        lead_time_advantage=8.0,
        key_metrics=["Installed capacity (GW)", "Data center PPA pipeline", "Nuclear fleet longevity"],
    ),
    InfraCompany(
        ticker="CEG",
        name="Constellation Energy",
        category="Power Generation",
        subcategory="Nuclear",
        physics_thesis=(
            "Largest US nuclear fleet. Microsoft signed a 20-year PPA. Nuclear provides "
            "the only scalable 24/7 zero-carbon power — uniquely suited for AI data center ESG mandates."
        ),
        constraint_controlled="US nuclear generation fleet — irreplaceable, 10+ years to add meaningfully",
        physics_score=9.0,
        cycle_stage="early",
        lead_time_advantage=10.0,
        key_metrics=["Nuclear capacity (GW)", "Corporate PPA pipeline", "Carbon-free premium pricing"],
    ),
    InfraCompany(
        ticker="NRG",
        name="NRG Energy",
        category="Power Generation",
        subcategory="Merchant Power",
        physics_thesis=(
            "Diversified gas + renewables fleet with retail energy arm. "
            "Growing large C&I (commercial/industrial) customer base including data center loads."
        ),
        constraint_controlled="Permitted gas generation capacity + retail customer relationships",
        physics_score=7.0,
        cycle_stage="early",
        lead_time_advantage=5.0,
        key_metrics=["Data center revenue mix", "New capacity additions", "Retail margin"],
    ),

    # ── PRE-CONNECTED POWER (CRYPTO MINERS → AI) ──────────────────────────────
    InfraCompany(
        ticker="CORZ",
        name="Core Scientific",
        category="Pre-Connected Power",
        subcategory="Crypto → AI Conversion",
        physics_thesis=(
            "Bitcoin miners own pre-permitted, pre-connected power infrastructure. "
            "CORZ is converting sites to host Nvidia GPUs under a CoreWeave contract. "
            "Building new data center power from scratch takes 5–7 years; CORZ has it now."
        ),
        constraint_controlled="Pre-connected power sites + permitting that took years to secure",
        physics_score=9.0,
        cycle_stage="early",
        lead_time_advantage=5.0,
        key_metrics=["HPC hosting MW contracted", "Power cost (¢/kWh)", "CoreWeave contract value"],
    ),
    InfraCompany(
        ticker="WULF",
        name="TeraWulf",
        category="Pre-Connected Power",
        subcategory="Crypto → AI Conversion",
        physics_thesis=(
            "Nuclear-powered Bitcoin miner (Nautilus facility, co-located at Susquehanna nuclear plant). "
            "Sub-$0.02/kWh power cost + zero-carbon is the ideal foundation for AI compute. "
            "Converting capacity to HPC/AI hosting."
        ),
        constraint_controlled="Nuclear-adjacent pre-connected power at industry-lowest cost",
        physics_score=8.5,
        cycle_stage="early",
        lead_time_advantage=6.0,
        key_metrics=["Power cost advantage vs. peers", "Nuclear partnership", "HPC capacity (MW)"],
    ),
    InfraCompany(
        ticker="IREN",
        name="Iris Energy",
        category="Pre-Connected Power",
        subcategory="Crypto → AI Conversion",
        physics_thesis=(
            "Canadian hydro-powered miner already running NVIDIA H100s for AI inference revenue. "
            "Cheap hydroelectric power + existing permitting = natural AI data center footprint."
        ),
        constraint_controlled="Cheap hydro-powered pre-connected data center sites",
        physics_score=8.0,
        cycle_stage="early",
        lead_time_advantage=4.0,
        key_metrics=["GPU cloud revenue growth", "Hydro power cost", "AI capacity (MW)"],
    ),
    InfraCompany(
        ticker="APLD",
        name="Applied Digital",
        category="Pre-Connected Power",
        subcategory="HPC Hosting",
        physics_thesis=(
            "Pivoted fully from crypto to HPC/AI hosting. Builds high-density AI data centers "
            "in power-rich markets (North Dakota). Long-term HPC contracts with AI companies."
        ),
        constraint_controlled="High-density AI compute hosting in power-rich, low-cost regions",
        physics_score=7.5,
        cycle_stage="early",
        lead_time_advantage=3.0,
        key_metrics=["HPC hosting revenue", "Contracted MW", "Power cost/kWh"],
    ),
    InfraCompany(
        ticker="RIOT",
        name="Riot Platforms",
        category="Pre-Connected Power",
        subcategory="Crypto Miner",
        physics_thesis=(
            "One of the largest Bitcoin miners; ~1 GW of pre-connected power in Texas. "
            "ERCOT power curtailment credits create unique cash flow. "
            "Infrastructure optionable for AI conversion."
        ),
        constraint_controlled="Pre-connected Texas power + ERCOT curtailment optionality",
        physics_score=7.0,
        cycle_stage="early",
        lead_time_advantage=3.0,
        key_metrics=["Installed power (MW)", "Curtailment revenue", "AI conversion timeline"],
    ),
    InfraCompany(
        ticker="MARA",
        name="Marathon Digital",
        category="Pre-Connected Power",
        subcategory="Crypto Miner",
        physics_thesis=(
            "Largest publicly traded Bitcoin miner by hashrate. "
            "Diversified global power infrastructure. "
            "Optionality to repurpose sites for AI is the physics arbitrage thesis."
        ),
        constraint_controlled="Global diversified power infrastructure + permitting pipeline",
        physics_score=6.5,
        cycle_stage="early",
        lead_time_advantage=2.0,
        key_metrics=["Hashrate (EH/s)", "Installed power (GW)", "AI optionality progress"],
    ),

    # ── GRID INFRASTRUCTURE ────────────────────────────────────────────────────
    InfraCompany(
        ticker="ETN",
        name="Eaton Corporation",
        category="Grid Infrastructure",
        subcategory="Electrical Equipment",
        physics_thesis=(
            "Makes power management, transformers, switchgear, and UPS systems. "
            "Every data center, EV charger, and renewable connection requires Eaton equipment. "
            "2–3 year order backlog; pricing power is exceptional."
        ),
        constraint_controlled="Electrical distribution equipment — physically can't build data centers without it",
        physics_score=8.0,
        cycle_stage="middle",
        lead_time_advantage=3.0,
        key_metrics=["Order backlog", "Data center segment revenue %", "Pricing power"],
    ),
    InfraCompany(
        ticker="GEV",
        name="GE Vernova",
        category="Grid Infrastructure",
        subcategory="Grid Equipment + Gas Turbines",
        physics_thesis=(
            "Makes gas turbines, wind turbines, and grid automation. "
            "Only company supplying both fast-ramp gas peakers (needed for AI firm power) "
            "and the grid automation equipment for smart interconnection. Multi-year order backlog."
        ),
        constraint_controlled="Gas turbine production capacity + grid automation — 3+ year delivery times",
        physics_score=8.5,
        cycle_stage="early",
        lead_time_advantage=4.0,
        key_metrics=["Turbine backlog ($B)", "Grid segment orders", "Electrification revenue"],
    ),
    InfraCompany(
        ticker="PWR",
        name="Quanta Services",
        category="Grid Infrastructure",
        subcategory="Electrical Contractor",
        physics_thesis=(
            "Largest electrical contractor in North America. Builds and maintains "
            "power transmission and distribution. Every new data center and generation project "
            "requires Quanta to physically connect it to the grid. $24B+ backlog."
        ),
        constraint_controlled="Skilled electrical labor and project management — the scarcest resource in grid buildout",
        physics_score=7.5,
        cycle_stage="early",
        lead_time_advantage=5.0,
        key_metrics=["Backlog ($24B+)", "Renewable + data center mix %", "Revenue per employee"],
    ),
    InfraCompany(
        ticker="HUBB",
        name="Hubbell",
        category="Grid Infrastructure",
        subcategory="Electrical Products",
        physics_thesis=(
            "Makes distribution transformers, wiring devices, and grid hardening products. "
            "Every electrical installation — data centers, grid upgrades, EV infrastructure — "
            "requires Hubbell components. Distribution transformer backlog is multi-year."
        ),
        constraint_controlled="Distribution transformers and wiring components — unavoidable physical requirement",
        physics_score=7.0,
        cycle_stage="middle",
        lead_time_advantage=2.0,
        key_metrics=["Utility segment growth", "Distribution transformer backlog", "Price/mix lift"],
    ),

    # ── COOLING & POWER MANAGEMENT ────────────────────────────────────────────
    InfraCompany(
        ticker="VRT",
        name="Vertiv Holdings",
        category="Cooling & Power Management",
        subcategory="Data Center Infrastructure",
        physics_thesis=(
            "H100/GB200 chips dissipate 700W–1,200W each — physics makes air cooling impossible beyond 50 kW/rack. "
            "Vertiv makes the liquid cooling and UPS systems that are physically mandatory for AI clusters. "
            "Order backlog exceeds 2 years."
        ),
        constraint_controlled="Liquid cooling and power management for AI data centers — no physical alternative",
        physics_score=9.0,
        cycle_stage="early",
        lead_time_advantage=4.0,
        key_metrics=["AI data center order mix", "Liquid cooling revenue", "Backlog ($B)"],
    ),
    InfraCompany(
        ticker="MOD",
        name="Modine Manufacturing",
        category="Cooling & Power Management",
        subcategory="Thermal Management",
        physics_thesis=(
            "Precision thermal management for data centers. "
            "AI chip thermal density forces adoption of advanced cooling solutions. "
            "Data center segment is now Modine's fastest-growing business line."
        ),
        constraint_controlled="Thermal management for high-density AI compute racks",
        physics_score=7.0,
        cycle_stage="early",
        lead_time_advantage=2.0,
        key_metrics=["Data center segment revenue %", "Thermal product margins", "Order growth"],
    ),

    # ── NUCLEAR REVIVAL ────────────────────────────────────────────────────────
    InfraCompany(
        ticker="CCJ",
        name="Cameco",
        category="Nuclear Revival",
        subcategory="Uranium Mining",
        physics_thesis=(
            "Largest publicly traded uranium miner. Nuclear renaissance — driven by AI power demand "
            "+ decarbonization — requires more uranium. Supply is structurally limited: mines take "
            "10+ years to bring online, and Kazakhstan controls 45% of world supply."
        ),
        constraint_controlled="Uranium mine supply — geologically scarce, decade-long production lead times",
        physics_score=8.5,
        cycle_stage="early",
        lead_time_advantage=10.0,
        key_metrics=["Uranium price ($/lb U3O8)", "Production volume (Mlb)", "Long-term contract %"],
    ),
    InfraCompany(
        ticker="OKLO",
        name="Oklo Inc.",
        category="Nuclear Revival",
        subcategory="Microreactor",
        physics_thesis=(
            "Factory-built microreactors (1.5–15 MW) deployable directly at data centers, "
            "bypassing the grid entirely — the ultimate physics arbitrage. "
            "Sam Altman-backed; NRC licensing in progress. Pure speculative optionality on nuclear SMRs."
        ),
        constraint_controlled="First-mover position in commercial microreactor licensing pipeline",
        physics_score=7.5,
        cycle_stage="early",
        lead_time_advantage=7.0,
        key_metrics=["NRC licensing milestones", "LOI pipeline (MW)", "Revenue timeline"],
    ),

    # ── CRITICAL MATERIALS ──────────────────────────────────────────────────────
    InfraCompany(
        ticker="FCX",
        name="Freeport-McMoRan",
        category="Critical Materials",
        subcategory="Copper Mining",
        physics_thesis=(
            "Copper is how electricity physically moves. Every data center, EV, and renewable installation "
            "needs copper. AI data centers use 4–5x more copper per MW than traditional data centers. "
            "Mine supply takes 10–20 years to develop; grade is declining globally."
        ),
        constraint_controlled="Copper mine production — geologically constrained with decade-long supply lead times",
        physics_score=8.0,
        cycle_stage="early",
        lead_time_advantage=10.0,
        key_metrics=["Copper production (kt)", "AISC ($/lb)", "Reserve grade trend"],
    ),
    InfraCompany(
        ticker="SCCO",
        name="Southern Copper",
        category="Critical Materials",
        subcategory="Copper Mining",
        physics_thesis=(
            "World's largest copper reserve base. Low-cost producer in Peru and Mexico. "
            "Structural demand from electrification + AI makes this a multi-decade story. "
            "Virtually irreplaceable reserve position."
        ),
        constraint_controlled="World's largest copper reserves — irreplaceable geographic asset",
        physics_score=8.0,
        cycle_stage="early",
        lead_time_advantage=10.0,
        key_metrics=["Copper reserves (billion tonnes)", "AISC ($/lb)", "Production growth (kt/yr)"],
    ),
]


class OpportunityScreener:
    def __init__(self):
        self.companies = COMPANY_DATABASE

    def as_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([
            {
                "Ticker": c.ticker,
                "Name": c.name,
                "Category": c.category,
                "Subcategory": c.subcategory,
                "Physics Score": c.physics_score,
                "Cycle Stage": c.cycle_stage,
                "Lead Time Adv (yrs)": c.lead_time_advantage,
                "Constraint Controlled": c.constraint_controlled,
                "Thesis": c.physics_thesis,
            }
            for c in self.companies
        ])

    def categories(self) -> List[str]:
        return sorted({c.category for c in self.companies})

    def top_by_score(self, n: int = 10) -> pd.DataFrame:
        return self.as_dataframe().nlargest(n, "Physics Score")
