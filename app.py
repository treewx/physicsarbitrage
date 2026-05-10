import os
import sys

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

sys.path.insert(0, os.path.dirname(__file__))

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
    st.title("⚛ Physics Arbitrage")
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
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🗺  Constraint Chain",
    "📈  Compute Demand",
    "📉  Wright's Law",
    "⚡  Energy Bottlenecks",
    "🔍  Opportunity Screener",
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

        # Color legend
        st.subheader("Node Types")
        for ntype, color in type_colors.items():
            st.markdown(
                f"<span style='color:{color}'>■</span> {ntype.capitalize()}",
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

        st.metric("GPT-4 training power", f"{gpt4_mw:.0f} MW", "~25,000 H100s × 90 days")
        st.metric("2025 frontier training", f"{f2025_mw:.0f} MW", f"~{f2025_chips:,} H100-equiv")
        st.metric("Compute doubling time", f"{doubling_months:.1f} months", "frontier runs, 2019–2025")
        st.metric("H100 chip power draw", "700 W each", f"× {mdl.PUE} PUE = facility overhead")

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
        st.caption("High severity + high investability = the sweet spot for physics arbitrage.")

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
                st.metric("Severity", f"{b.constraint_severity}/10")
                st.metric("Lead Time to Fix", f"{b.lead_time_years:.0f} yrs")
                if b.current_capacity_gw > 0:
                    st.metric("Current Build Rate", f"{b.current_capacity_gw:.0f} GW/yr")


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
        cats = st.multiselect("Category", screener.categories(), default=screener.categories())
    with col_f2:
        min_score = st.slider("Min Physics Score", 1.0, 10.0, 7.0, 0.5)
    with col_f3:
        stages = st.multiselect("Cycle Stage", ["early", "middle", "late", "mature"], default=["early", "middle"])

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
            with col_nums:
                st.markdown(f"<span class='{cls}'>{score:.1f}/10</span>", unsafe_allow_html=True)
                st.metric("Lead Time Advantage", f"{row['Lead Time Adv (yrs)']:.1f} yrs")
                st.markdown(f"**Cycle Stage:** `{row['Cycle Stage']}`")
                if show_live and "Price" in row and pd.notna(row.get("Price")):
                    st.metric("Price", row["Price"])
                    st.metric("Market Cap", f"${row.get('Mkt Cap ($B)', '?')}B")
                    st.metric("1Y Performance", row.get("1Y Perf", "N/A"))

    # Summary table
    st.divider()
    st.subheader("Screener Table")
    table_cols = ["Ticker", "Name", "Category", "Physics Score", "Lead Time Adv (yrs)", "Cycle Stage", "Constraint Controlled"]
    if show_live and "Price" in df_filtered.columns:
        table_cols += ["Price", "Mkt Cap ($B)", "Fwd P/E", "EV/EBITDA", "Rev Growth", "1Y Perf"]
    avail = [c for c in table_cols if c in df_filtered.columns]
    st.dataframe(df_filtered[avail], use_container_width=True, hide_index=True)

    st.divider()
    st.caption(
        "⚠️ **Disclaimer:** This tool is for educational and analytical purposes only — "
        "it does not constitute investment advice. Physics arbitrage analysis identifies structural "
        "opportunities but does not account for valuation, execution risk, or timing. "
        "Always perform your own due diligence."
    )
