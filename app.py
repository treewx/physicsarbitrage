import base64
import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))


def _commit_csv_to_github(csv_local_path: str) -> bool:
    """
    Push the updated companies.csv to GitHub via the Contents API.
    Returns True on success, False if no token is configured.
    Raises requests.HTTPError on API failure.
    """
    try:
        token  = st.secrets.get("GITHUB_TOKEN")  or os.environ.get("GITHUB_TOKEN", "")
        repo   = st.secrets.get("GITHUB_REPO")   or os.environ.get("GITHUB_REPO",  "treewx/physicsarbitrage")
        branch = st.secrets.get("GITHUB_BRANCH") or os.environ.get("GITHUB_BRANCH", "master")
    except Exception:
        token  = os.environ.get("GITHUB_TOKEN", "")
        repo   = os.environ.get("GITHUB_REPO",  "treewx/physicsarbitrage")
        branch = os.environ.get("GITHUB_BRANCH", "master")

    if not token:
        return False

    file_path = "data/companies.csv"
    api_url   = f"https://api.github.com/repos/{repo}/contents/{file_path}"
    headers   = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}

    # Fetch current SHA (required by the API to update an existing file)
    r = requests.get(api_url, headers=headers, params={"ref": branch}, timeout=15)
    r.raise_for_status()
    sha = r.json()["sha"]

    with open(csv_local_path, "r", encoding="utf-8") as f:
        encoded = base64.b64encode(f.read().encode("utf-8")).decode("utf-8")

    r2 = requests.put(api_url, headers=headers, timeout=15, json={
        "message": "Add company via Claude evaluator",
        "content": encoded,
        "sha":     sha,
        "branch":  branch,
    })
    r2.raise_for_status()
    return True

from data.fetchers import fetch_stock_data, fetch_price_history, market_data_to_df
from models.compute import ComputeDemandModel
from models.energy import EnergyBottleneckModel, BOTTLENECKS
from models.screening import OpportunityScreener
from models.wright_law import WrightLawModel, TECHNOLOGY_DATABASE

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Physics Arbitrage Scanner",
    page_icon="⚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .block-container { padding-top: 5rem; }
    .stTabs [data-baseweb="tab"] { font-size: 14px; font-weight: 600; }
    .tag {
        display: inline-block; padding: 2px 8px; border-radius: 4px;
        font-size: 0.75em; margin: 2px; background: #2d2d4e; color: #ccc;
    }
    .thesis-text { font-size: 0.85em; color: #bbb; line-height: 1.55; }
    .score-high { color: #4CAF50; font-weight: bold; font-size: 1.3em; }
    .score-med  { color: #FFC107; font-weight: bold; font-size: 1.3em; }
    .score-low  { color: #42A5F5; font-weight: bold; font-size: 1.3em; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        '<h1 style="margin-bottom:0" title="Physics arbitrage: the strategy of identifying '
        'the hard physical constraints a major technology wave will inevitably hit — energy, '
        'materials, manufacturing, labour — then investing in companies that already control '
        'those constrained assets before the market appreciates their scarcity. '
        'Coined and popularised by Leopold Aschenbrenner (Situational Awareness LP).">⚛ Physics Arbitrage</h1>',
        unsafe_allow_html=True,
    )
    st.caption("Aschenbrenner / Situational Awareness framework")
    st.divider()

    st.divider()
    show_live = st.toggle(
        "Fetch Live Market Data",
        value=False,
        help="Pull real-time prices and fundamentals via yfinance. Slower; requires internet.",
    )

    st.divider()
    st.markdown("""
**The Framework**

1. Pick a major tech wave
2. Trace it upstream through physics
3. Find who *already owns* the scarce bottleneck
4. Estimate whether it's priced in
5. Bet before the crowd — exit when it's obvious

*"Go upstream all the way to the physical constraints that every AI company will eventually slam into."* — Aschenbrenner
    """)

# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🗺  Constraint Chain",
    "📈  Compute Demand",
    "📉  Wright's Law",
    "⚡  Energy Bottlenecks",
    "🔍  Opportunity Screener",
    "⏱  Sequencing",
])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — CONSTRAINT CHAIN
# ═══════════════════════════════════════════════════════════════
with tab1:
    st.header("Physics Constraint Chain")
    technology_wave = st.selectbox(
        "Technology Wave",
        ["AI / AGI Scaling", "Electric Vehicles", "Green Hydrogen", "Nuclear Revival", "Robotics"],
    )
    st.markdown(
        "Trace the selected technology wave upstream through its physical dependencies. "
        "The most upstream, hardest-to-replicate resource is where the arbitrage lives."
    )

    CHAINS = {
        "AI / AGI Scaling": {
            "blurb": (
                "AI training compute doubles roughly every 6 months. Each doubling requires more "
                "chips, more power, more cooling — in that physical order. Capital can be raised "
                "in days; power infrastructure takes years."
            ),
            "nodes": [
                (0, "AGI / LLMs",             "demand",         "Exponentially growing frontier model training and inference"),
                (1, "GPU Clusters",            "compute",        "H100/GB200 clusters — TSMC CoWoS packaging is the chokepoint"),
                (1, "Electric Power",          "energy",         "700W–1,200W per chip × millions of chips = GW-scale demand"),
                (2, "Grid Connection",         "infrastructure", "5–7 year interconnection queue in the US; pre-connected sites command premiums"),
                (2, "Power Generation",        "infrastructure", "Gas, nuclear, renewables — 24/7 firm power especially scarce"),
                (2, "Liquid Cooling",          "infrastructure", "Mandatory for >50 kW/rack AI compute; air cooling is physically insufficient"),
                (3, "Transformers",            "materials",      "1–2 year delivery times; no domestic US large-power-transformer manufacturer"),
                (3, "Copper Wiring",           "materials",      "AI data centers use 4–5× more copper per MW than traditional data centers"),
                (3, "Skilled Electricians",    "labor",          "300k+ additional needed by 2030; 4–5 year apprenticeship pipeline"),
                (4, "Copper Mining",           "upstream",       "10–20 year mine development timelines; global ore grade in structural decline"),
                (4, "Uranium Mining",          "upstream",       "Nuclear baseload for firm 24/7 power requires more uranium supply"),
            ],
            "bottleneck": "Grid interconnection + Pre-connected power infrastructure",
            "plays": ["Bloom Energy (BE)", "Core Scientific (CORZ)", "TeraWulf (WULF)", "Vistra (VST)", "Constellation (CEG)", "Vertiv (VRT)", "GE Vernova (GEV)"],
        },
        "Electric Vehicles": {
            "blurb": "EVs require 5–10× more copper per vehicle and ~80 kWh of battery per car. Mass adoption triggers commodity supercycles in lithium, cobalt, copper, and rare earths.",
            "nodes": [
                (0, "EV Adoption",             "demand",         "Regulatory mandates + cost crossover driving mass-market transition"),
                (1, "Battery Packs",           "compute",        "LFP/NMC chemistry — 60–100 kWh per vehicle"),
                (1, "Electric Motors",         "infrastructure", "Permanent-magnet motors require neodymium and dysprosium"),
                (2, "Lithium",                 "materials",      "Hard-rock and brine sources — 5–10 year project lead times"),
                (2, "Copper",                  "materials",      "EVs use 4× more copper than ICE vehicles"),
                (2, "Charging Infrastructure", "infrastructure", "Grid upgrades required for mass simultaneous charging"),
                (3, "Cobalt / Nickel",         "upstream",       "Battery cathode materials — DRC supply concentration risk"),
                (3, "Rare Earth Elements",     "upstream",       "China controls 85%+ of REE processing globally"),
            ],
            "bottleneck": "Lithium supply + battery gigafactory capacity",
            "plays": ["Freeport (FCX)", "Albemarle (ALB)", "Livent (LTHM)", "MP Materials (MP)", "Piedmont Lithium (PLL)"],
        },
        "Green Hydrogen": {
            "blurb": "Electrolyzers split water with renewable electricity to produce zero-carbon hydrogen. Scaling requires GW-scale dedicated renewable power and electrolyzer manufacturing at 100× current volumes.",
            "nodes": [
                (0, "Green H2 Demand",         "demand",         "Industrial decarbonization: steel, ammonia, shipping all need green H2"),
                (1, "Electrolyzers",           "compute",        "Alkaline/PEM/SOEC — following Wright's Law but still early on the curve"),
                (1, "Renewable Power",         "energy",         "~50–60 kWh of electricity consumed per kg of H2 produced"),
                (2, "Grid Capacity",           "infrastructure", "GW-scale dedicated renewable generation required"),
                (2, "H2 Storage/Distribution", "infrastructure", "Pipelines, compression, storage — largely absent today"),
                (3, "Platinum / Iridium",      "materials",      "PEM electrolyzers require platinum-group metals — finite supply"),
                (3, "Nickel",                  "upstream",       "Alkaline electrolyzers use nickel electrodes"),
            ],
            "bottleneck": "Electrolyzer manufacturing capacity + renewable power supply",
            "plays": ["Plug Power (PLUG)", "Bloom Energy (BE)", "Air Products (APD)", "Linde (LIN)"],
        },
        "Nuclear Revival": {
            "blurb": "AI data centers + decarbonization mandates create unprecedented demand for 24/7 carbon-free baseload. Nuclear is uniquely suited but has decade-long construction lead times — making uranium supply and existing plants extremely valuable.",
            "nodes": [
                (0, "Clean Firm Power Demand", "demand",         "AI + ESG mandates require 24/7 zero-carbon baseload at scale"),
                (1, "Nuclear Reactors",        "compute",        "Existing fleet + new SMRs — 10-year construction timelines for new builds"),
                (1, "Uranium Fuel",            "energy",         "Enriched uranium fuel — supply chain almost entirely non-Western"),
                (2, "Uranium Mining",          "infrastructure", "Kazakhstan (45% of world supply), Canada, Australia"),
                (2, "Uranium Conversion",      "infrastructure", "U3O8 → UF6 — only a handful of facilities globally"),
                (2, "Enrichment Capacity",     "infrastructure", "LEU/HALEU enrichment — Urenco, Centrus; Russia sanctioned"),
                (3, "Nuclear Engineers",       "labor",          "Multi-decade training pipeline; retirement wave underway"),
                (3, "Nuclear-Grade Steel",     "materials",      "Stringent quality assurance; very few qualified suppliers"),
            ],
            "bottleneck": "Uranium enrichment capacity + nuclear-grade manufacturing",
            "plays": ["Cameco (CCJ)", "Centrus (LEU)", "Oklo (OKLO)", "Constellation (CEG)", "Vistra (VST)"],
        },
        "Robotics": {
            "blurb": "Physical AI (humanoid and industrial robots) requires precision actuators, torque-dense motors, and edge AI chips. The upstream bottleneck is rare-earth magnets — 85% of which are processed in China.",
            "nodes": [
                (0, "Robot Adoption",          "demand",         "Labor shortages + AI capability driving humanoid + industrial demand"),
                (1, "Actuators / Motors",      "compute",        "High-torque electric motors with sub-millisecond feedback control"),
                (1, "Edge AI Chips",           "compute",        "Power-efficient inference silicon for mobile/embedded robots"),
                (2, "Rare Earth Magnets",      "materials",      "Neodymium-iron-boron magnets for motors — China controls supply"),
                (2, "Precision Manufacturing", "infrastructure", "CNC machining, bearings, harmonic drives — limited high-precision capacity"),
                (3, "Dysprosium / Nd",         "upstream",       "Rare earth elements — China processes 85%+ of global REE output"),
            ],
            "bottleneck": "Rare-earth magnet supply chain + precision actuator manufacturing",
            "plays": ["MP Materials (MP)", "Rockwell Automation (ROK)", "Cognex (CGNX)", "Nvidia (NVDA)"],
        },
    }

    chain = CHAINS[technology_wave]
    col_chain, col_side = st.columns([3, 1])

    type_colors = {
        "demand": "#EF5350",
        "compute": "#AB47BC",
        "energy": "#FFA726",
        "infrastructure": "#42A5F5",
        "materials": "#66BB6A",
        "upstream": "#26C6DA",
        "labor": "#EC407A",
    }

    with col_chain:
        st.subheader(f"Chain: {technology_wave}")
        st.markdown(chain["blurb"])
        st.write("")

        for depth, node, ntype, desc in chain["nodes"]:
            color = type_colors.get(ntype, "#888")
            arrow = "◆" if depth == 0 else ("→" * depth)
            indent = "&nbsp;" * depth * 10
            st.markdown(
                f"{indent}"
                f"<span style='color:{color}; font-size:1.05em;'>{arrow}</span> "
                f"**{node}** "
                f"<span class='tag'>{ntype}</span><br>"
                f"{indent}&nbsp;&nbsp;&nbsp;<span class='thesis-text'>{desc}</span>",
                unsafe_allow_html=True,
            )
            st.write("")

    with col_side:
        st.subheader("Key Bottleneck")
        st.info(f"**{chain['bottleneck']}**")

        st.subheader("Aschenbrenner-Style Plays")
        for play in chain["plays"]:
            st.markdown(f"- {play}")

        st.divider()

        # Color legend with tooltips
        st.subheader("Node Types")
        node_explanations = {
            "demand":         "The end-market driving the need (e.g. AI model training). This is where the investment thesis starts — everything upstream of demand is what physics arbitrage is hunting.",
            "compute":        "The processing hardware required (GPUs, TPUs, ASICs). Usually the first bottleneck the market discovers and prices in.",
            "energy":         "The raw electricity needed to run the compute. Often underappreciated until compute is already fully priced in — this is the current bottleneck for AI.",
            "infrastructure": "Physical facilities and connections (data centres, grid interconnections, cooling). Takes years to permit and build — hence the value of pre-existing sites.",
            "materials":      "Physical inputs required at scale: copper wiring, transformer steel, rare-earth magnets. Supply is geologically constrained and cannot be rushed.",
            "upstream":       "The furthest-back resources in the chain (mining, raw extraction). Longest lead times, last to be priced in by markets — the most contrarian and highest-asymmetry positions.",
            "labor":          "Skilled workers needed: electricians, nuclear engineers, grid technicians. A constraint that capital alone cannot fix quickly — training pipelines take years.",
        }
        for ntype, color in type_colors.items():
            tooltip = node_explanations.get(ntype, "")
            st.markdown(
                f'<span style="color:{color}" title="{tooltip}">■</span> '
                f'<span title="{tooltip}">{ntype.capitalize()}</span>',
                unsafe_allow_html=True,
            )

    # Historical analogies
    st.divider()
    st.subheader("Historical Physics Arbitrage Playbook")
    st.markdown("Every major technology wave has had upstream physical bottlenecks that were obvious *in retrospect* but invisible to most investors in real time.")

    analogies = pd.DataFrame([
        {
            "Era": "Internet (1995–2005)",
            "Technology Wave": "World Wide Web",
            "Physical Constraint": "Fiber bandwidth, rack space, power",
            "Early Plays": "Cisco (CSCO), dark fiber, colo REITs",
            "Lesson": "Distinguish the constraint (fiber) from the application (dot-coms). Fiber built; apps mostly failed.",
        },
        {
            "Era": "Shale Revolution (2005–2015)",
            "Technology Wave": "Horizontal drilling + fracking",
            "Physical Constraint": "Drill bits, fracking sand, pipeline takeaway",
            "Early Plays": "Halliburton (HAL), U.S. Silica (SLCA), midstream MLPs",
            "Lesson": "Service companies capture value before commodity price collapse. Know when to rotate.",
        },
        {
            "Era": "EV Transition (2020s)",
            "Technology Wave": "Battery electric vehicles",
            "Physical Constraint": "Lithium, cobalt, charging grid",
            "Early Plays": "Albemarle (ALB), MP Materials (MP), ChargePoint",
            "Lesson": "Commodity supercycles attract capital fast. Supply eventually catches up — timing matters.",
        },
        {
            "Era": "AI Infrastructure (2023–?)",
            "Technology Wave": "AI / AGI Scaling",
            "Physical Constraint": "Power, pre-connected sites, cooling",
            "Early Plays": "Bloom Energy (BE), crypto miners → AI, Vistra (VST), Vertiv (VRT)",
            "Lesson": "Grid bottleneck takes 5–7 years to fix. Miners already have the plug in the wall.",
        },
    ])
    st.dataframe(analogies, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# TAB 2 — COMPUTE DEMAND
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.header("AI Compute & Power Demand")
    st.markdown(
        "Frontier AI training compute has doubled roughly every 6 months since 2020. "
        "Each doubling requires proportionally more chips, power, and cooling. "
        "Here's where the physics hits the infrastructure."
    )

    mdl = ComputeDemandModel()
    hist_df = mdl.historical_compute_df()

    col_hist, col_kpi = st.columns([3, 1])

    with col_hist:
        # Log-scale compute growth with trend line
        years = hist_df["year"].values
        log_flops = np.log10(hist_df["flops"].values)
        z = np.polyfit(years, log_flops, 1)
        p = np.poly1d(z)
        doubling_months = 12.0 / (z[0] * np.log2(10))

        trend_x = np.linspace(years[0], years[-1] + 1.5, 200)

        fig_hist = go.Figure()
        fig_hist.add_trace(go.Scatter(
            x=trend_x, y=10 ** p(trend_x),
            mode="lines", name=f"Trend ({doubling_months:.1f}-mo doubling)",
            line=dict(color="#FF7043", dash="dash", width=1.5),
        ))
        fig_hist.add_trace(go.Scatter(
            x=hist_df["year"], y=hist_df["flops"],
            mode="markers+lines",
            name="Training Compute (FLOPs)",
            marker=dict(size=9, color="#AB47BC"),
            line=dict(color="#AB47BC", width=2),
            text=hist_df["name"],
            hovertemplate="<b>%{text}</b><br>Year: %{x:.1f}<br>FLOPs: %{y:.2e}<extra></extra>",
        ))
        fig_hist.update_layout(
            title="Frontier AI Training Compute (FLOPs)",
            xaxis_title="Year",
            yaxis_title="Training FLOPs (log scale)",
            yaxis_type="log",
            template="plotly_dark",
            height=370,
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_kpi:
        st.subheader("Physics Checks")
        gpt4_mw = mdl.training_power_mw(2.15e25, 90)
        f2025_mw = mdl.training_power_mw(5e26, 90)
        f2025_chips = mdl.chips_needed(5e26, 90)

        st.metric("GPT-4 training power", f"{gpt4_mw:.0f} MW", "~25,000 H100s × 90 days",
                  help="Total facility power (compute + cooling overhead) needed to train a GPT-4 class model in 90 days. For comparison, a large hospital uses about 5–10 MW.")
        st.metric("2025 frontier training", f"{f2025_mw:.0f} MW", f"~{f2025_chips:,} H100-equiv",
                  help="Estimated power for a single frontier training run at today's scale. Each new generation requires roughly 4× more compute than the last.")
        st.metric("Compute doubling time", f"{doubling_months:.1f} months", "frontier runs, 2019–2025",
                  help="How frequently the total compute used in the largest training runs has doubled. This is faster than Moore's Law and is the root cause of the power demand explosion.")
        st.metric("H100 chip power draw", "700 W each", f"× {mdl.PUE} PUE = facility overhead",
                  help=f"Each NVIDIA H100 GPU consumes 700W. PUE (Power Usage Effectiveness) of {mdl.PUE} accounts for cooling and other overhead — so every 700W of chips needs {700*mdl.PUE:.0f}W of total facility power.")

    # Industry power projection
    st.divider()
    st.subheader("Projected AI Industry Power Demand")
    time_horizon = st.slider("Projection horizon (years)", 3, 12, 8, key="horizon_compute")

    demand_df = mdl.project_industry_power_gw(years=time_horizon)

    fig_power = go.Figure()
    fig_power.add_trace(go.Scatter(
        x=demand_df["year"], y=demand_df["total_power_gw"],
        fill="tozeroy", fillcolor="rgba(255,87,34,0.18)",
        name="Total AI Power (training + inference)",
        line=dict(color="#FF5722", width=2),
    ))
    fig_power.add_trace(go.Scatter(
        x=demand_df["year"], y=demand_df["training_power_gw"],
        fill="tozeroy", fillcolor="rgba(171,71,188,0.18)",
        name="Training Only",
        line=dict(color="#AB47BC", width=2),
    ))
    fig_power.add_hline(y=100, line_dash="dot", line_color="#aaa",
                        annotation_text="US Nuclear Fleet today (100 GW)")
    fig_power.add_hline(y=1100, line_dash="dot", line_color="#FFA726",
                        annotation_text="US Total Generating Capacity (~1,100 GW)")
    fig_power.update_layout(
        xaxis_title="Year",
        yaxis_title="Power (GW)",
        template="plotly_dark",
        height=420,
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
    )
    st.plotly_chart(fig_power, use_container_width=True)

    st.caption(
        "⚠️ Projections assume current scaling trends persist. Even conservative scenarios "
        "imply massive new power infrastructure. The question is not *whether* AI grows — "
        "it is *who already has the infrastructure* before the crowd arrives."
    )

    # Power table
    st.divider()
    st.subheader("Power Requirement by Model Generation")
    st.dataframe(mdl.power_table(), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════
# TAB 3 — WRIGHT'S LAW
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.header("Wright's Law — Technology Cost Curves")
    st.markdown(
        "**Wright's Law (1936):** For every doubling of cumulative production, cost falls by "
        "a fixed percentage. This creates predictable cost trajectories — and predictable "
        "demand inflection points when technologies cross key price thresholds."
    )
    st.latex(r"\text{Cost}(N) = \text{Cost}_0 \cdot \left(\frac{N}{N_0}\right)^{-\alpha}, \quad \alpha = -\log_2(1 - \text{LR})")

    col_chart, col_ctrl = st.columns([3, 1])

    with col_ctrl:
        selected = st.multiselect(
            "Technologies to display",
            [t.name for t in TECHNOLOGY_DATABASE],
            default=["Solar PV", "Lithium-Ion Batteries", "AI Chips ($/TFLOP/s)"],
        )

        st.divider()
        st.subheader("Custom Tech")
        c_name = st.text_input("Name", "My Technology")
        c_lr = st.slider("Learning Rate (%)", 5, 50, 20) / 100.0
        c_init = st.number_input("Initial Cost", value=1000.0, min_value=0.01)
        c_curr = st.number_input("Current Cost", value=100.0, min_value=0.01)
        c_unit = st.text_input("Unit", "$/unit")

    with col_chart:
        fig_wl = go.Figure()

        for tech in TECHNOLOGY_DATABASE:
            if tech.name not in selected:
                continue
            wl = WrightLawModel(tech)
            curve = wl.historical_curve()

            fig_wl.add_trace(go.Scatter(
                x=curve["volume"], y=curve["cost"],
                mode="lines", name=tech.name,
                line=dict(color=tech.color, width=2),
                hovertemplate=f"<b>{tech.name}</b><br>Volume: %{{x:.2f}}<br>Cost: %{{y:.3f}} {tech.unit}<extra></extra>",
            ))
            if tech.current_volume and tech.current_cost:
                fig_wl.add_trace(go.Scatter(
                    x=[tech.current_volume], y=[tech.current_cost],
                    mode="markers", showlegend=False,
                    marker=dict(size=12, color=tech.color, symbol="star"),
                    hovertemplate=f"<b>{tech.name} — Today</b><br>Volume: {tech.current_volume:.0f}<br>Cost: {tech.current_cost:.2f} {tech.unit}<extra></extra>",
                ))

        fig_wl.update_layout(
            title="Technology Cost vs. Cumulative Production",
            xaxis_title="Cumulative Production (log scale)",
            yaxis_title="Cost per Unit (log scale)",
            xaxis_type="log",
            yaxis_type="log",
            template="plotly_dark",
            height=430,
        )
        st.plotly_chart(fig_wl, use_container_width=True)

    # Summary table
    st.divider()
    st.subheader("Learning Rate Reference Table")
    lr_rows = []
    for tech in TECHNOLOGY_DATABASE:
        wl = WrightLawModel(tech)
        drop_10x = (1 - wl.cost_at_volume(tech.initial_volume * 10) / tech.initial_cost) * 100
        if tech.current_volume and tech.current_cost:
            yrs_half = wl.years_to_cost(tech.current_cost * 0.5, tech.current_volume, tech.annual_growth_rate)
            yrs_half_str = f"{yrs_half:.1f}"
        else:
            yrs_half_str = "N/A"
        lr_rows.append({
            "Technology": tech.name,
            "Learning Rate": f"{tech.learning_rate*100:.0f}%",
            "Cost Drop per Doubling": f"{tech.learning_rate*100:.0f}%",
            "Cost Drop (10× production)": f"{drop_10x:.0f}%",
            "Years to 50% Cost Reduction": yrs_half_str,
            "Volume Growth/yr": f"{tech.annual_growth_rate*100:.0f}%",
            "Unit": tech.unit,
        })
    st.dataframe(pd.DataFrame(lr_rows), use_container_width=True, hide_index=True)

    # Forward projection
    st.divider()
    st.subheader("Forward Cost Projection")
    col_proj_ctrl, col_proj_horizon = st.columns([2, 1])
    with col_proj_ctrl:
        proj_name = st.selectbox("Project forward:", [t.name for t in TECHNOLOGY_DATABASE])
    with col_proj_horizon:
        time_horizon = st.slider("Years to project", 3, 12, 8, key="horizon_wright")
    proj_tech = next(t for t in TECHNOLOGY_DATABASE if t.name == proj_name)
    wl_fwd = WrightLawModel(proj_tech)

    if proj_tech.current_volume and proj_tech.current_cost:
        proj_df = wl_fwd.project_forward(time_horizon, proj_tech.current_volume, proj_tech.annual_growth_rate)

        fig_fwd = make_subplots(rows=1, cols=2, subplot_titles=("Projected Cost", "Cumulative Cost Reduction (%)"))
        fig_fwd.add_trace(go.Scatter(
            x=proj_df["year"], y=proj_df["cost"],
            name="Cost", line=dict(color=proj_tech.color, width=2)
        ), row=1, col=1)
        fig_fwd.add_trace(go.Scatter(
            x=proj_df["year"], y=proj_df["cost_reduction_pct"],
            name="% vs Today", fill="tozeroy",
            line=dict(color="#FFA726", width=2),
            fillcolor="rgba(255,167,38,0.18)",
        ), row=1, col=2)
        fig_fwd.update_yaxes(title_text=proj_tech.unit, row=1, col=1)
        fig_fwd.update_yaxes(title_text="% below today's cost", row=1, col=2)
        fig_fwd.update_layout(template="plotly_dark", height=280, showlegend=False)
        st.plotly_chart(fig_fwd, use_container_width=True)

        final_cost = proj_df["cost"].iloc[-1]
        final_reduction = -proj_df["cost_reduction_pct"].iloc[-1]
        st.info(
            f"At a **{proj_tech.learning_rate*100:.0f}% learning rate** and "
            f"**{proj_tech.annual_growth_rate*100:.0f}%/yr volume growth**, "
            f"**{proj_name}** could reach **{final_cost:.2f} {proj_tech.unit}** "
            f"by {int(proj_df['year'].iloc[-1])} — "
            f"a **{final_reduction:.0f}% reduction** from today's cost."
        )


# ═══════════════════════════════════════════════════════════════
# TAB 4 — ENERGY BOTTLENECKS
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.header("Energy Infrastructure Bottlenecks")
    st.markdown(
        "AI companies can raise billions in hours. Power infrastructure takes years to build. "
        "The grid is the binding constraint — and Aschenbrenner's entire thesis rests on this asymmetry."
    )

    energy_mdl = EnergyBottleneckModel()

    col_radar, col_gap = st.columns(2)

    with col_radar:
        st.subheader("Bottleneck Map")
        radar_df = energy_mdl.bottleneck_radar_df()

        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df["severity"],
            theta=radar_df["bottleneck"],
            fill="toself", name="Constraint Severity",
            line=dict(color="#FF5722"),
            fillcolor="rgba(255,87,34,0.20)",
        ))
        fig_radar.add_trace(go.Scatterpolar(
            r=radar_df["investability"],
            theta=radar_df["bottleneck"],
            fill="toself", name="Investability (10 − lead-time)",
            line=dict(color="#42A5F5"),
            fillcolor="rgba(66,165,245,0.20)",
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
            template="plotly_dark", height=420,
        )
        st.plotly_chart(fig_radar, use_container_width=True)
        st.caption(
            "**Severity** = how critical is this bottleneck (1–10). "
            "**Investability** = 10 minus the lead time in years — high investability means "
            "the bottleneck is real *and* you can actually buy something that benefits from it today. "
            "Top-right of both axes = the sweet spot."
        )

    with col_gap:
        st.subheader("AI Power: Demand vs. New Supply")
        time_horizon = st.slider("Projection horizon (years)", 3, 12, 8, key="horizon_energy")
        gap_df = energy_mdl.demand_supply_gap(base_ai_demand_gw=15, years=time_horizon)

        fig_gap = go.Figure()
        fig_gap.add_trace(go.Bar(
            x=gap_df["year"], y=gap_df["gap_gw"],
            name="Unmet Demand (GW)", marker_color="#EF5350",
        ))
        fig_gap.add_trace(go.Scatter(
            x=gap_df["year"], y=gap_df["demand_gw"],
            name="AI Demand (GW)", line=dict(color="#FF7043", dash="dash"),
        ))
        fig_gap.add_trace(go.Scatter(
            x=gap_df["year"], y=gap_df["supply_gw"],
            name="New AI-Ready Supply (GW)", line=dict(color="#66BB6A", dash="dash"),
        ))
        fig_gap.update_layout(
            xaxis_title="Year", yaxis_title="GW",
            template="plotly_dark", height=400,
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01),
        )
        st.plotly_chart(fig_gap, use_container_width=True)

    # Detailed bottleneck cards
    st.divider()
    st.subheader("Bottleneck Details")

    for b in BOTTLENECKS:
        icon = "🔴" if b.constraint_severity >= 9 else ("🟡" if b.constraint_severity >= 7.5 else "🟢")
        with st.expander(f"{icon} {b.name}  —  Severity: {b.constraint_severity}/10  |  Fix Lead Time: {b.lead_time_years:.0f} yrs"):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(b.description)
                st.markdown("**Investable sectors:**")
                for sector in b.investable_sectors:
                    st.markdown(f"  - {sector}")
            with col_b:
                st.metric("Severity", f"{b.constraint_severity}/10",
                          help="How critical is this bottleneck to the AI infrastructure buildout? 9–10 = existential constraint that physically cannot be bypassed in the near term. 7–8 = serious friction that adds years and cost. Below 7 = significant but has partial workarounds.")
                st.metric("Lead Time to Fix", f"{b.lead_time_years:.0f} yrs",
                          help="How many years would it take to meaningfully expand supply of this resource, even with unlimited capital committed today? Longer = more durable scarcity = higher value for companies that already have it.")
                if b.current_capacity_gw > 0:
                    st.metric("Current Build Rate", f"{b.current_capacity_gw:.0f} GW/yr",
                              help="How much new capacity is being added in this category each year right now (GW). Compare this against the AI power demand projections in the Compute Demand tab to see the size of the gap.")


# ═══════════════════════════════════════════════════════════════
# TAB 5 — OPPORTUNITY SCREENER
# ═══════════════════════════════════════════════════════════════
with tab5:
    st.header("Opportunity Screener")
    st.markdown(
        "Public companies that control physical bottlenecks in the AI infrastructure buildout. "
        "Scored by how central they are to the physics constraint, how early we are in the cycle, "
        "and how long before competition can replicate their position."
    )

    screener = OpportunityScreener()
    df_all = screener.as_dataframe()

    # Filters
    col_f1, col_f2, col_f3 = st.columns(3)
    with col_f1:
        cats = st.multiselect(
            "Category", screener.categories(), default=screener.categories(), key="cat_filter",
        )
    with col_f2:
        min_score = st.slider("Min Physics Score", 1.0, 10.0, 1.0, 0.5)
    with col_f3:
        stages = st.multiselect(
            "Cycle Stage", ["early", "middle", "late", "mature"],
            default=["early", "middle", "late", "mature"], key="stage_filter",
        )

    df_filtered = df_all[
        df_all["Category"].isin(cats) &
        (df_all["Physics Score"] >= min_score) &
        df_all["Cycle Stage"].isin(stages)
    ].sort_values("Physics Score", ascending=False).reset_index(drop=True)

    st.markdown(f"**{len(df_filtered)} companies match your filters**")

    # Bubble chart
    fig_bubble = px.scatter(
        df_filtered,
        x="Lead Time Adv (yrs)",
        y="Physics Score",
        color="Category",
        text="Ticker",
        size="Physics Score",
        size_max=28,
        template="plotly_dark",
        title="Physics Score vs. Lead-Time Advantage",
        height=420,
        hover_data={"Name": True, "Subcategory": True, "Cycle Stage": True},
    )
    fig_bubble.update_traces(textposition="top center")
    st.plotly_chart(fig_bubble, use_container_width=True)

    # Optional live data
    if show_live and len(df_filtered) > 0:
        tickers = df_filtered["Ticker"].tolist()
        with st.spinner(f"Fetching live market data for {len(tickers)} tickers via yfinance…"):
            raw = fetch_stock_data(tickers, period="1y")
        market_df = market_data_to_df(raw)
        if not market_df.empty:
            df_filtered = df_filtered.merge(market_df, on="Ticker", how="left")

    # Company cards
    st.divider()
    for _, row in df_filtered.iterrows():
        score = row["Physics Score"]
        if score >= 9:
            icon, cls = "🔴", "score-high"
        elif score >= 7.5:
            icon, cls = "🟡", "score-med"
        else:
            icon, cls = "🟢", "score-low"

        with st.expander(
            f"{icon} [{row['Ticker']}] {row['Name']}  —  "
            f"Physics Score: {score:.1f}/10  |  {row['Category']}  |  Stage: {row['Cycle Stage'].upper()}"
        ):
            col_thesis, col_nums = st.columns([3, 1])
            with col_thesis:
                st.markdown(f"**Constraint Controlled:** {row['Constraint Controlled']}")
                st.markdown(f"**Physics Thesis:**")
                st.markdown(f"<span class='thesis-text'>{row['Thesis']}</span>", unsafe_allow_html=True)
                _reasoning = row.get("reasoning", "")
                if pd.notna(_reasoning) and str(_reasoning).strip():
                    st.markdown("**Claude's Scoring Rationale:**")
                    st.caption(str(_reasoning))
            with col_nums:
                st.markdown(f"<span class='{cls}'>{score:.1f}/10</span>", unsafe_allow_html=True)
                st.metric("Lead Time Advantage", f"{row['Lead Time Adv (yrs)']:.1f} yrs",
                          help="How many years it would take a new competitor to replicate this company's physical position from scratch. A mine takes 10–20 years; a pre-connected power site takes 5–7 years; electrical equipment takes 2–3 years. The longer this is, the more durable the moat.")
                st.markdown(f"**Cycle Stage:** `{row['Cycle Stage']}`")
                if show_live and "Price" in row and pd.notna(row.get("Price")):
                    st.metric("Price", row["Price"])
                    st.metric("Market Cap", f"${row.get('Mkt Cap ($B)', '?')}B")
                    st.metric("1Y Performance", row.get("1Y Perf", "N/A"))

    # Formula explainer
    st.divider()
    with st.expander("ℹ️ How is Physics Score calculated?"):
        from models.screening import OWNERSHIP_WT, SUPPLY_WT
        st.markdown(f"""
**Formula:** `Physics Score = Ownership Score × {OWNERSHIP_WT} + Supply Constraint Score × {SUPPLY_WT}`

| Input | Scale | What it measures |
|---|---|---|
| **Ownership Score** | 1 – 10 | Does this company directly *own or control* the bottleneck asset? 10 = sole owner of something scarce (e.g. a uranium mine, a pre-connected power site). 5 = important player but with real competition. |
| **Supply Constraint Score** | 1 – 10 | How hard is it for new supply to come online, even with unlimited capital? 10 = geologically or physically impossible near-term (mine development, nuclear construction). 6 = 2–5 years. 4 = 1–2 years. |

Both scores live in **`data/companies.csv`** and can be edited directly in Excel or Google Sheets.
The Physics Score column in the table is computed automatically every time the app loads.
        """)

    # Summary table
    st.subheader("Screener Table")
    table_cols = ["Ticker", "Name", "Category", "Physics Score", "Ownership Score", "Supply Constraint Score", "Lead Time Adv (yrs)", "Cycle Stage", "Constraint Controlled"]
    if show_live and "Price" in df_filtered.columns:
        table_cols += ["Price", "Mkt Cap ($B)", "Fwd P/E", "EV/EBITDA", "Rev Growth", "1Y Perf"]
    avail = [c for c in table_cols if c in df_filtered.columns]
    st.dataframe(
        df_filtered[avail],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Physics Score": st.column_config.NumberColumn(
                "Physics Score",
                help=f"Computed as: Ownership × {OWNERSHIP_WT} + Supply Constraint × {SUPPLY_WT}. Edit the two input columns in data/companies.csv to change this score.",
                format="%.1f",
            ),
            "Ownership Score": st.column_config.NumberColumn(
                "Ownership Score",
                help="How directly does this company OWN or CONTROL the bottleneck asset? 10 = sole owner of something scarce (a mine, a pre-connected power site). 7 = primary provider with some competition. 5 = important player but substitutable. Edit in data/companies.csv.",
                format="%.0f",
            ),
            "Supply Constraint Score": st.column_config.NumberColumn(
                "Supply Constraint Score",
                help="How hard is it for new supply to respond to price signals, even with unlimited capital? 10 = geologically or physically impossible near-term (mine development, nuclear build). 6 = 2-5 years. 4 = 1-2 years. Edit in data/companies.csv.",
                format="%.0f",
            ),
            "Lead Time Adv (yrs)": st.column_config.NumberColumn(
                "Lead Time Adv (yrs)",
                help="How many years it would take a new competitor to replicate this company's physical position from scratch. A copper mine takes 10–20 years. A pre-connected data centre site takes 5–7 years. Electrical equipment takes 2–3 years. Longer = more durable competitive moat.",
                format="%.1f",
            ),
            "Cycle Stage": st.column_config.TextColumn(
                "Cycle Stage",
                help="Where is the market in recognising this bottleneck asset? 'Early' = the AI connection is not yet consensus — most of the upside is still ahead. 'Middle' = awareness growing, partially priced in. 'Late' = widely known, mostly priced in. 'Mature' = fully priced in, look to rotate out.",
            ),
            "Constraint Controlled": st.column_config.TextColumn(
                "Constraint Controlled",
                help="The specific physical resource or infrastructure that this company controls — the actual bottleneck asset that cannot be quickly replicated. This is the core of the physics arbitrage thesis: whoever owns the constraint captures the value.",
            ),
            "Category": st.column_config.TextColumn(
                "Category",
                help="The type of physical infrastructure this company operates in: Power Generation = makes electricity; Pre-Connected Power = already has grid connections (crypto miners converting to AI); Grid Infrastructure = equipment and contractors that connect power to consumers; Cooling = manages heat from AI chips; Nuclear = zero-carbon baseload; Critical Materials = raw inputs like copper and uranium.",
            ),
        },
    )

    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This tool is for educational and analytical purposes only — "
        "it does not constitute investment advice. Physics arbitrage analysis identifies structural "
        "opportunities but does not account for valuation, execution risk, or timing. "
        "Always perform your own due diligence."
    )

    # ── Shared API key for both Claude features below ────────────
    _api_key = None
    try:
        _api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    if not _api_key:
        _api_key = os.environ.get("ANTHROPIC_API_KEY")

    _no_key_msg = (
        "No Anthropic API key found.  \n"
        "**Streamlit Cloud:** go to *Settings → Secrets* and add `ANTHROPIC_API_KEY = \"sk-ant-…\"`.  \n"
        "**Local:** set the `ANTHROPIC_API_KEY` environment variable before running the app."
    )

    # ── Add a single company ──────────────────────────────────────
    st.divider()
    with st.expander("➕ Add a Company with Claude AI"):
        st.markdown(
            "Enter a ticker and company name. Claude will fetch basic company data from Yahoo Finance, "
            "evaluate it against the physics arbitrage rubric, and propose scores. "
            "Review the result before confirming — it gets saved to `data/companies.csv` and "
            "appears in the screener immediately."
        )

        if not _api_key:
            st.warning(_no_key_msg)
        else:
            col_et, col_en = st.columns([1, 2])
            with col_et:
                eval_ticker = st.text_input(
                    "Ticker", placeholder="e.g. GEV", key="eval_ticker",
                ).strip().upper()
            with col_en:
                eval_name = st.text_input(
                    "Company Name", placeholder="e.g. GE Vernova", key="eval_name",
                ).strip()

            already_in = df_all["Ticker"].str.upper().tolist()

            if st.button("🤖 Evaluate with Claude", disabled=not (eval_ticker and eval_name)):
                if eval_ticker in already_in:
                    st.warning(f"**{eval_ticker}** is already in the screener. Remove it from `data/companies.csv` first if you want to re-evaluate.")
                else:
                    with st.spinner(f"Claude is researching {eval_ticker} ({eval_name})…"):
                        try:
                            from data.evaluator import evaluate_company
                            st.session_state["eval_result"] = evaluate_company(eval_ticker, eval_name, _api_key)
                            st.session_state.pop("eval_error", None)
                        except Exception as exc:
                            st.session_state["eval_error"] = str(exc)
                            st.session_state.pop("eval_result", None)

            if "eval_error" in st.session_state:
                st.error(f"Evaluation failed: {st.session_state['eval_error']}")

            if "eval_result" in st.session_state:
                r = st.session_state["eval_result"]
                ctx = r.get("yf_context", {})

                from models.screening import OWNERSHIP_WT, SUPPLY_WT
                physics_preview = round(
                    r["ownership_score"] * OWNERSHIP_WT + r["supply_response_score"] * SUPPLY_WT, 1
                )

                st.success("Claude's evaluation — review before saving:")

                col_sc, col_th = st.columns([1, 2])
                with col_sc:
                    st.metric("Physics Score", f"{physics_preview}/10",
                              help=f"= Ownership × {OWNERSHIP_WT} + Supply Constraint × {SUPPLY_WT}")
                    st.metric("Ownership Score", f"{r['ownership_score']}/10",
                              help="Does this company own or control a physical bottleneck asset?")
                    st.metric("Supply Constraint Score", f"{r['supply_response_score']}/10",
                              help="How fast can new supply come online even with unlimited capital?")
                    st.markdown(f"**Category:** {r['category']}")
                    st.markdown(f"**Subcategory:** {r['subcategory']}")
                    st.markdown(f"**Cycle Stage:** `{r['cycle_stage']}`")
                    st.markdown(f"**Lead Time Advantage:** {r['lead_time_advantage_yrs']} yrs")

                with col_th:
                    if ctx:
                        st.markdown(
                            f"**Sector:** {ctx.get('sector', 'N/A')}  |  "
                            f"**Industry:** {ctx.get('industry', 'N/A')}  |  "
                            f"**Market Cap:** ${ctx.get('market_cap_b', 0)}B"
                        )
                    st.markdown(f"**Constraint Controlled:** {r['constraint_controlled']}")
                    st.markdown("**Physics Thesis:**")
                    st.markdown(f"<span class='thesis-text'>{r['physics_thesis']}</span>", unsafe_allow_html=True)
                    st.markdown(f"**Key Metrics:** {r['key_metrics']}")
                    st.markdown("**Claude's Reasoning:**")
                    st.info(r["reasoning"])

                col_ok, col_no = st.columns(2)
                with col_ok:
                    if st.button("✅ Confirm & Add to Screener", type="primary"):
                        _csv_path = os.path.join(os.path.dirname(__file__), "data", "companies.csv")
                        _new_row = {k: v for k, v in r.items() if k not in ("yf_context",)}
                        _existing = pd.read_csv(_csv_path)
                        pd.concat([_existing, pd.DataFrame([_new_row])], ignore_index=True).to_csv(_csv_path, index=False)
                        # Commit back to GitHub so the addition survives redeploys
                        _gh_ok = False
                        try:
                            _gh_ok = _commit_csv_to_github(_csv_path)
                        except Exception as _gh_err:
                            st.warning(f"Saved for this session but GitHub sync failed: {_gh_err}")
                        st.session_state.pop("eval_result", None)
                        st.session_state.pop("cat_filter", None)
                        st.session_state.pop("stage_filter", None)
                        if _gh_ok:
                            st.success(f"**{r['ticker']}** added and committed to GitHub — change is permanent.")
                        else:
                            st.info(
                                f"**{r['ticker']}** added for this session. "
                                "To make it permanent, add `GITHUB_TOKEN` to your Streamlit secrets "
                                "(Settings → Secrets)."
                            )
                        st.rerun()
                with col_no:
                    if st.button("🗑 Discard"):
                        st.session_state.pop("eval_result", None)
                        st.rerun()

    # ── Batch re-evaluation ───────────────────────────────────────
    st.divider()
    with st.expander("🔄 Batch Re-evaluate All Companies with Claude"):
        st.markdown(
            "Re-score every company in the database using the current Claude rubric so all scores "
            "are consistent and comparable. Claude evaluates each company in turn, then shows you "
            "a side-by-side comparison of old vs new scores. Nothing is saved until you confirm."
        )

        if not _api_key:
            st.warning(_no_key_msg)
        else:
            from data.evaluator import evaluate_company as _eval_company
            from models.screening import OWNERSHIP_WT as _OWT, SUPPLY_WT as _SWT

            _n = len(df_all)
            st.markdown(f"**{_n} companies** will be re-evaluated — takes roughly **{_n * 4}–{_n * 6} seconds**.")

            if "batch_results" not in st.session_state:
                if st.button(f"🚀 Start Batch Re-evaluation ({_n} companies)"):
                    _batch_ok, _batch_err = [], []
                    _prog = st.progress(0.0)
                    _status = st.empty()
                    for _i, (_, _row) in enumerate(df_all.iterrows()):
                        _t, _nm = _row["Ticker"], _row["Name"]
                        _status.markdown(f"Evaluating **{_t}** ({_i + 1} / {_n})…")
                        try:
                            _batch_ok.append(_eval_company(_t, _nm, _api_key))
                        except Exception as _e:
                            _batch_err.append({"ticker": _t, "name": _nm, "error": str(_e)})
                        _prog.progress((_i + 1) / _n)
                    _status.empty()
                    _prog.empty()
                    st.session_state["batch_results"] = _batch_ok
                    st.session_state["batch_errors"] = _batch_err
                    st.rerun()

            if "batch_results" in st.session_state:
                _results = st.session_state["batch_results"]
                _errors  = st.session_state.get("batch_errors", [])

                if _errors:
                    st.error(f"{len(_errors)} companies failed:")
                    for _e in _errors:
                        st.caption(f"  • {_e['ticker']}: {_e['error']}")

                # Build comparison table
                _cmp_rows = []
                for _r in _results:
                    _old = df_all[df_all["Ticker"] == _r["ticker"]]
                    _old_own  = int(_old["Ownership Score"].iloc[0])       if not _old.empty else None
                    _old_sup  = int(_old["Supply Constraint Score"].iloc[0]) if not _old.empty else None
                    _old_phys = float(_old["Physics Score"].iloc[0])       if not _old.empty else None
                    _new_phys = round(_r["ownership_score"] * _OWT + _r["supply_response_score"] * _SWT, 1)
                    _cmp_rows.append({
                        "Ticker":       _r["ticker"],
                        "Name":         _r["name"],
                        "Old Own.":     _old_own,
                        "New Own.":     _r["ownership_score"],
                        "Δ Own.":       _r["ownership_score"] - _old_own if _old_own is not None else None,
                        "Old Supply":   _old_sup,
                        "New Supply":   _r["supply_response_score"],
                        "Δ Supply":     _r["supply_response_score"] - _old_sup if _old_sup is not None else None,
                        "Old Physics":  _old_phys,
                        "New Physics":  _new_phys,
                        "Δ Physics":    round(_new_phys - _old_phys, 1) if _old_phys is not None else None,
                        "New Category": _r["category"],
                        "Reasoning":    _r.get("reasoning", ""),
                    })

                _cmp_df = pd.DataFrame(_cmp_rows)
                st.dataframe(
                    _cmp_df,
                    use_container_width=True,
                    hide_index=True,
                    column_config={
                        "Δ Own.":    st.column_config.NumberColumn(format="%+.0f"),
                        "Δ Supply":  st.column_config.NumberColumn(format="%+.0f"),
                        "Δ Physics": st.column_config.NumberColumn(format="%+.1f"),
                        "Reasoning": st.column_config.TextColumn(width="large"),
                    },
                )

                _col_yes, _col_no2 = st.columns(2)
                with _col_yes:
                    if st.button("✅ Confirm & Replace All Scores", type="primary"):
                        _csv_path = os.path.join(os.path.dirname(__file__), "data", "companies.csv")
                        _new_rows = [{k: v for k, v in _r.items() if k != "yf_context"} for _r in _results]
                        pd.DataFrame(_new_rows).to_csv(_csv_path, index=False)
                        _gh_ok = False
                        try:
                            _gh_ok = _commit_csv_to_github(_csv_path)
                        except Exception as _ge:
                            st.warning(f"Saved locally but GitHub sync failed: {_ge}")
                        for _k in ("batch_results", "batch_errors", "cat_filter", "stage_filter"):
                            st.session_state.pop(_k, None)
                        if _gh_ok:
                            st.success("All scores updated and committed to GitHub!")
                        else:
                            st.success("All scores updated for this session.")
                        st.rerun()
                with _col_no2:
                    if st.button("🗑 Discard Batch Results"):
                        st.session_state.pop("batch_results", None)
                        st.session_state.pop("batch_errors", None)
                        st.rerun()


# ═══════════════════════════════════════════════════════════════
# TAB 6 — SEQUENCING
# ═══════════════════════════════════════════════════════════════
with tab6:
    st.header("Sequencing — Getting the Timing Right")
    st.markdown("""
    Aschenbrenner's most important insight isn't *which* physical bottlenecks matter —
    it's *when* to be in each one. Markets discover bottlenecks **sequentially**.
    Once a constraint becomes consensus, the multiple expansion is largely done.
    The money is always in the **next** undiscovered bottleneck up the chain.

    > *"The sequencing matters as much as the thesis. NVIDIA was right —
    but by late 2023 it was consensus. The power bottleneck was the next shoe to drop,
    and the market hadn't connected those dots yet."*
    """)

    # ── Data — loaded from data/sequencing.csv ──────────────────
    _seq_path = os.path.join(os.path.dirname(__file__), "data", "sequencing.csv")
    _seq_raw = pd.read_csv(_seq_path)
    _signal_colors = {"Buy": "#4CAF50", "Hold": "#FFC107", "Reduce": "#EF5350"}
    _seq_raw["signal_color"] = _seq_raw["signal"].map(_signal_colors).fillna("#888")
    SEQ = _seq_raw.to_dict("records")

    # ── legacy fallback key kept for reference (not used) ───────
    if False:
        {
            "category": "Chips / Compute",
            "phase": "Saturating",
            "phase_num": 5,
            "discovery_yr": 2020.5,
            "consensus_yr": 2023.5,
            "market_awareness": 90,
            "upside_remaining": 2,
            "physics_conviction": 10,
            "signal": "Reduce / Avoid new positions",
            "signal_color": "#EF5350",
            "tickers": "NVDA, TSM, AVGO, ASML",
            "representative_return": "NVDA +800% from 2022 lows",
            "window": "Passed",
            "thesis": (
                "AI training requires GPUs. NVIDIA had a near-monopoly on training chips; "
                "TSMC on manufacturing them. Both ran 5–10x as the market connected AI demand to compute supply."
            ),
            "why_done": (
                "Supply is catching up. AMD, Google TPU, AWS Trainium, and Meta's MTIA are eroding the monopoly. "
                "Forward multiples are extremely stretched. The constraint is still real — but it's priced in."
            ),
            "what_to_watch": "AMD market share gains; custom silicon adoption rates by hyperscalers",
        },
        {
            "category": "Power Generation",
            "phase": "Awakening → Consensus",
            "phase_num": 3,
            "discovery_yr": 2023.5,
            "consensus_yr": 2025.8,
            "market_awareness": 65,
            "upside_remaining": 5,
            "physics_conviction": 10,
            "signal": "Hold — selectively add on dips",
            "signal_color": "#FFC107",
            "tickers": "BE, VST, CEG, NRG",
            "representative_return": "VST +300%, Bloom Energy +239% (2025)",
            "window": "Now → 12–18 months",
            "thesis": (
                "Data centers need 24/7 firm power. Grid connection takes 5–7 years. "
                "Generators with existing permitted capacity are the only near-term solution. "
                "Bloom Energy's fuel cells bypass the queue entirely — Aschenbrenner's largest position at $855M."
            ),
            "why_done": (
                "Not fully done — but getting there. The hyperscaler PPA thesis is now widely understood. "
                "Easy money is behind us; remaining upside is real but requires more patience and selectivity."
            ),
            "what_to_watch": "Hyperscaler PPA announcements; grid interconnection reform; natural gas prices",
        },
        {
            "category": "Pre-Connected Power (Miners → AI)",
            "phase": "Early Awakening",
            "phase_num": 2,
            "discovery_yr": 2024.0,
            "consensus_yr": 2026.5,
            "market_awareness": 42,
            "upside_remaining": 7,
            "physics_conviction": 9,
            "signal": "Buy — early in discovery cycle",
            "signal_color": "#4CAF50",
            "tickers": "CORZ, WULF, IREN, APLD, RIOT",
            "representative_return": "CORZ restructured; WULF +200% (2024); still early",
            "window": "Now → 2 years",
            "thesis": (
                "Bitcoin miners have pre-permitted, pre-connected power that took 3–5 years to secure. "
                "Converting a mining site to AI hosting takes 6–12 months. "
                "Building equivalent infrastructure from scratch takes 5–7 years and costs 3–5× more. "
                "The 'plug already in the wall' is the asset — not the Bitcoin mining business."
            ),
            "why_done": (
                "Still early. Most generalist investors haven't connected crypto miners to AI infrastructure. "
                "The conversion thesis is playing out (CORZ/CoreWeave deal) but is not yet consensus."
            ),
            "what_to_watch": "HPC contract MW signed; power cost vs. peers; CoreWeave/Microsoft hosting deals",
        },
        {
            "category": "Data Center Cooling",
            "phase": "Early Awakening",
            "phase_num": 2,
            "discovery_yr": 2023.8,
            "consensus_yr": 2026.0,
            "market_awareness": 50,
            "upside_remaining": 6,
            "physics_conviction": 9,
            "signal": "Buy — physics is non-negotiable",
            "signal_color": "#4CAF50",
            "tickers": "VRT, MOD",
            "representative_return": "Vertiv +250% (2024); still order backlog growing",
            "window": "Now → 18 months",
            "thesis": (
                "H100/GB200 chips dissipate 700–1,200W each. Air cooling physically cannot handle "
                ">50 kW/rack. Liquid cooling is not optional for next-gen AI clusters — it is a "
                "law of thermodynamics. Vertiv's order backlog exceeds 2 years."
            ),
            "why_done": (
                "Partially discovered. Vertiv has run but the backlog keeps growing faster than the stock. "
                "Smaller players (Modine) are less discovered. The cooling constraint compounds as chips get denser."
            ),
            "what_to_watch": "Liquid cooling attach rate in new builds; Vertiv backlog growth; GB200 cluster deployments",
        },
        {
            "category": "Grid Infrastructure",
            "phase": "Discovery",
            "phase_num": 1,
            "discovery_yr": 2024.5,
            "consensus_yr": 2027.5,
            "market_awareness": 28,
            "upside_remaining": 8,
            "physics_conviction": 8,
            "signal": "Buy — sub-consensus, long runway",
            "signal_color": "#4CAF50",
            "tickers": "GEV, ETN, PWR, HUBB",
            "representative_return": "GEV +80% since spin-off; ETN +60% — but the AI connection not widely made",
            "window": "1–3 years",
            "thesis": (
                "Every new power plant, data center, and EV charger needs electrical equipment and "
                "contractors to connect it to the grid. Large power transformers have 1–2 year delivery times "
                "with no US domestic manufacturer. Skilled electricians are the binding constraint — "
                "300k+ needed by 2030 with a 4-year training pipeline."
            ),
            "why_done": (
                "Not yet widely discovered in the context of AI. These companies are seen as 'boring industrials.' "
                "The AI connection (data centers need grid buildout → grid buildout needs Eaton/Quanta) "
                "has not become consensus. This is a 1–3 year story."
            ),
            "what_to_watch": "Transformer backlog growth; Quanta data center mix; transmission capex announcements",
        },
        {
            "category": "Nuclear Revival",
            "phase": "Discovery",
            "phase_num": 1,
            "discovery_yr": 2024.0,
            "consensus_yr": 2028.0,
            "market_awareness": 35,
            "upside_remaining": 8,
            "physics_conviction": 7,
            "signal": "Buy — long duration, highest conviction for patient capital",
            "signal_color": "#4CAF50",
            "tickers": "CCJ, CEG, OKLO, LEU",
            "representative_return": "CCJ +150% (2023–24); OKLO speculative early-stage",
            "window": "2–5 years",
            "thesis": (
                "AI data centers need 24/7 carbon-free power. Wind and solar are intermittent. "
                "Nuclear is the only scalable answer. The US fleet is aging; new SMRs are 5–10 years away. "
                "Uranium supply is structurally tight — Kazakhstan controls 45% and mines take 10+ years to open."
            ),
            "why_done": (
                "Partially discovered via the Microsoft/Three Mile Island PPA announcement. "
                "But the broader AI → nuclear connection is not yet consensus. "
                "This is a long-duration position requiring patience — the payoff is 2–5 years out."
            ),
            "what_to_watch": "Corporate nuclear PPA announcements; NRC licensing pace; HALEU enrichment capacity",
        },
        {
            "category": "Copper & Critical Materials",
            "phase": "Pre-Discovery",
            "phase_num": 0,
            "discovery_yr": 2026.0,
            "consensus_yr": 2029.5,
            "market_awareness": 15,
            "upside_remaining": 9,
            "physics_conviction": 8,
            "signal": "Early accumulation — longest lead time, most asymmetric",
            "signal_color": "#4CAF50",
            "tickers": "FCX, SCCO, MP, ALB",
            "representative_return": "FCX flat-to-down in 2024 — AI thesis not yet connected to copper",
            "window": "3–6 years",
            "thesis": (
                "Copper is how electricity physically moves. AI data centers use 4–5× more copper per MW "
                "than traditional data centers. Mine development takes 10–20 years. Global ore grade is "
                "in structural decline. The market has not yet connected the AI buildout thesis to copper demand — "
                "this is the last domino to fall."
            ),
            "why_done": (
                "Barely started. Most copper bulls are focused on EVs, not AI. "
                "The AI → power → copper connection is the least-appreciated link in the chain. "
                "This is the most contrarian, longest-duration, highest-asymmetry position in the sequence."
            ),
            "what_to_watch": "Data center copper intensity research; mine pipeline; China EV demand (competing thesis)",
        }

    df_seq = pd.DataFrame(SEQ)

    # ── 1. Discovery Quadrant ────────────────────────────────────
    st.subheader("The Discovery Quadrant")
    st.markdown(
        "**X axis:** How much has the market already priced in the AI demand thesis for this category?  \n"
        "**Y axis:** How much upside remains if the physics thesis fully plays out?  \n"
        "The **top-left** is where you want to be: high upside, low awareness."
    )

    col_quad, col_legend = st.columns([3, 1])

    with col_quad:
        fig_quad = go.Figure()

        # Quadrant shading
        fig_quad.add_shape(type="rect", x0=0, x1=50, y0=5, y1=10,
                           fillcolor="rgba(76,175,80,0.08)", line_width=0)
        fig_quad.add_shape(type="rect", x0=50, x1=100, y0=5, y1=10,
                           fillcolor="rgba(255,193,7,0.08)", line_width=0)
        fig_quad.add_shape(type="rect", x0=0, x1=50, y0=0, y1=5,
                           fillcolor="rgba(66,165,245,0.06)", line_width=0)
        fig_quad.add_shape(type="rect", x0=50, x1=100, y0=0, y1=5,
                           fillcolor="rgba(239,83,80,0.08)", line_width=0)

        # Quadrant labels
        for x, y, label in [(25, 9.5, "BUY ZONE"), (75, 9.5, "HOLD / TRIM"),
                             (25, 0.5, "WATCH"), (75, 0.5, "AVOID")]:
            fig_quad.add_annotation(x=x, y=y, text=f"<b>{label}</b>",
                                    showarrow=False, font=dict(size=11, color="#555"),
                                    opacity=0.6)

        fig_quad.add_trace(go.Scatter(
            x=df_seq["market_awareness"],
            y=df_seq["upside_remaining"],
            mode="markers+text",
            text=df_seq["category"].apply(lambda x: x.split(" (")[0].split(" &")[0]),
            textposition="top center",
            marker=dict(
                size=df_seq["physics_conviction"] * 4,
                color=df_seq["signal_color"],
                line=dict(width=1.5, color="white"),
                opacity=0.85,
            ),
            customdata=df_seq[["phase", "tickers", "window"]],
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Market Awareness: %{x}%<br>"
                "Upside Remaining: %{y}/10<br>"
                "Phase: %{customdata[0]}<br>"
                "Key Tickers: %{customdata[1]}<br>"
                "Window: %{customdata[2]}<extra></extra>"
            ),
        ))

        fig_quad.add_vline(x=50, line_dash="dash", line_color="#555", line_width=1)
        fig_quad.add_hline(y=5, line_dash="dash", line_color="#555", line_width=1)

        fig_quad.update_layout(
            xaxis=dict(title="Market Awareness (% of thesis priced in)", range=[0, 100]),
            yaxis=dict(title="Upside Remaining (if thesis plays out)", range=[0, 10]),
            template="plotly_dark",
            height=430,
            showlegend=False,
        )
        st.plotly_chart(fig_quad, use_container_width=True)

    with col_legend:
        st.markdown("**Bubble size** = physics conviction score")
        st.markdown("**Bubble color** = rotation signal")
        st.markdown("")
        st.markdown("🟢 **Buy** — early, underpriced")
        st.markdown("🟡 **Hold** — partially priced in")
        st.markdown("🔴 **Reduce** — largely priced in")
        st.divider()
        st.markdown("**The sequence so far:**")
        for row in SEQ:
            icon = "✅" if row["phase_num"] >= 4 else ("🔄" if row["phase_num"] == 3 else "⏳")
            st.markdown(f"{icon} {row['category'].split(' (')[0].split(' &')[0]}")

    # ── 2. Phase Timeline ────────────────────────────────────────
    st.divider()
    st.subheader("Bottleneck Discovery Timeline")
    st.markdown(
        "When did (or will) each category get discovered by the market? "
        "The red line is today. Everything to the right is still ahead of us."
    )

    today = 2025.4

    fig_timeline = go.Figure()

    for i, row in enumerate(reversed(SEQ)):
        idx = len(SEQ) - 1 - i
        done = row["consensus_yr"] < today
        bar_color = "#EF5350" if done else ("#4CAF50" if row["market_awareness"] < 50 else "#FFC107")

        fig_timeline.add_trace(go.Bar(
            x=[row["consensus_yr"] - row["discovery_yr"]],
            y=[row["category"].split(" (")[0]],
            base=[row["discovery_yr"]],
            orientation="h",
            marker_color=bar_color,
            marker_opacity=0.75,
            name=row["phase"],
            showlegend=False,
            customdata=[[row["phase"], row["window"], row["signal"]]],
            hovertemplate=(
                f"<b>{row['category']}</b><br>"
                "Discovery window: %{base:.1f} → %{x:.1f}<br>"
                "Phase: %{customdata[0]}<br>"
                "Timing window: %{customdata[1]}<br>"
                "Signal: %{customdata[2]}<extra></extra>"
            ),
        ))

    # Today marker
    fig_timeline.add_vline(x=today, line_color="#FF5722", line_width=2.5,
                           annotation_text="TODAY", annotation_position="top")

    fig_timeline.update_layout(
        xaxis=dict(title="Year", range=[2019, 2031]),
        yaxis=dict(title=""),
        template="plotly_dark",
        height=320,
        barmode="overlay",
        margin=dict(l=220),
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

    # ── 3. Where to be NOW ──────────────────────────────────────
    st.divider()
    st.subheader("Where to Be Right Now")

    buy_now = [r for r in SEQ if r["signal_color"] == "#4CAF50"]
    hold = [r for r in SEQ if r["signal_color"] == "#FFC107"]
    avoid = [r for r in SEQ if r["signal_color"] == "#EF5350"]

    col_buy, col_hold, col_avoid = st.columns(3)

    with col_buy:
        st.markdown("### 🟢 Buy")
        for r in buy_now:
            st.markdown(f"**{r['category'].split(' (')[0]}**  \n"
                        f"_{r['tickers']}_  \n"
                        f"Window: {r['window']}")
            st.write("")

    with col_hold:
        st.markdown("### 🟡 Hold / Selective")
        for r in hold:
            st.markdown(f"**{r['category']}**  \n"
                        f"_{r['tickers']}_  \n"
                        f"Window: {r['window']}")
            st.write("")

    with col_avoid:
        st.markdown("### 🔴 Reduce / Avoid New")
        for r in avoid:
            st.markdown(f"**{r['category']}**  \n"
                        f"_{r['tickers']}_  \n"
                        f"Window: {r['window']}")
            st.write("")

    # ── 4. Detail cards ─────────────────────────────────────────
    st.divider()
    st.subheader("Full Sequencing Detail")

    for row in SEQ:
        color = row["signal_color"]
        icon = "🟢" if color == "#4CAF50" else ("🟡" if color == "#FFC107" else "🔴")
        with st.expander(
            f"{icon} {row['category']}  —  {row['phase']}  |  "
            f"Market Awareness: {row['market_awareness']}%  |  Signal: {row['signal']}"
        ):
            col_a, col_b = st.columns([3, 1])
            with col_a:
                st.markdown(f"**The thesis:**  \n{row['thesis']}")
                st.write("")
                st.markdown(f"**Why it's at this stage:**  \n{row['why_done']}")
                st.write("")
                st.markdown(f"**What to watch:**  \n{row['what_to_watch']}")
            with col_b:
                st.metric("Market Awareness", f"{row['market_awareness']}%",
                          help="Estimate of how much of the physics-driven demand thesis for this category has already been priced into the relevant stocks. 0% = market hasn't connected this to AI at all. 100% = fully priced in, easy money gone. The lower this is relative to physics conviction, the more attractive the opportunity.")
                st.metric("Upside Remaining", f"{row['upside_remaining']}/10",
                          help="Subjective estimate of how much further these stocks could run if the full physics thesis plays out and is not yet priced in. 9–10 = very large potential move still ahead. 1–2 = most of the move has already happened.")
                st.metric("Physics Conviction", f"{row['physics_conviction']}/10",
                          help="How certain is it, based on physical laws and infrastructure economics, that this will be a genuine bottleneck? 10 = thermodynamic or geological certainty (e.g. you cannot cool H100s without liquid cooling). 7 = strong but with some substitutability or timeline uncertainty.")
                st.markdown(f"**Timing window:** {row['window']}")
                st.markdown(f"**Key tickers:** `{row['tickers']}`")
                st.markdown(f"**Representative return:** {row['representative_return']}")

    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** Sequencing analysis reflects one interpretation of market cycle positioning "
        "as of mid-2025. It is not investment advice. Timing is inherently uncertain — "
        "cycles can compress or extend significantly. Always do your own research."
    )
