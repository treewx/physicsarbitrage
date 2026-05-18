"""
Claude-powered candidate idea generator for the Physics Arbitrage screener.

Given a strategy type and an AI stack layer, asks Claude to brainstorm
6-8 under-the-radar publicly-traded companies that fit the physics
arbitrage profile for that combination.
"""
import json
import re

import anthropic


STRATEGY_DESCRIPTIONS = {
    "Asset Control": (
        "Companies that directly OWN or CONTROL a scarce physical asset — a mine, "
        "a pre-connected power site, a nuclear plant, a pre-permitted grid connection. "
        "Value comes from ownership of the asset itself, not from manufacturing or services."
    ),
    "Mandatory Supplier": (
        "Companies that manufacture the only or dominant physical component without which "
        "a bottleneck layer cannot be unlocked at scale. They don't own the bottleneck — "
        "they supply the key that opens it. Examples: ASML for EUV lithography machines; "
        "Lumentum for the laser chips inside AI optical transceivers."
    ),
    "Conversion Optionality": (
        "Companies with existing physical infrastructure that can be repurposed for AI use "
        "faster and cheaper than new infrastructure can be built from scratch. The arbitrage "
        "is in the time gap between conversion (months) and new-build (years). "
        "Example: crypto miners converting pre-connected power sites to AI hosting."
    ),
    "Input Supply": (
        "Companies supplying the upstream inputs — fuel, raw materials, industrial gases, "
        "chemicals, water — that feed the AI infrastructure stack one or two layers above "
        "them. Often overlooked because they are not directly classified in the 'AI' sector."
    ),
    "Special Situation": (
        "Contrarian situations where the market is materially mispricing an asset's "
        "strategic value to the AI buildout. Could be sector misclassification, recent "
        "bad news creating an entry point, or the market simply not having made the "
        "connection to AI demand yet."
    ),
}

LAYER_DESCRIPTIONS = {
    "Compute": (
        "GPU/accelerator chips — designing, manufacturing, advanced packaging (CoWoS), "
        "high-bandwidth memory (HBM) stacking on AI chips"
    ),
    "Power Generation": (
        "Producing the electricity data centres need 24/7 — gas turbines, fuel cells, "
        "nuclear reactors, diesel/gas generators, modular power units"
    ),
    "Grid & Transmission": (
        "Moving power from generator to data centre — large power transformers, "
        "transmission cables, switchgear, protection relays, electrical contractors"
    ),
    "Cooling": (
        "Managing heat from high-density AI racks — liquid cooling distribution units (CDUs), "
        "chillers, dry coolers, heat exchangers, cold plates, immersion cooling fluids"
    ),
    "Optical Interconnects": (
        "Moving data between chips, servers and racks at scale — optical transceivers, "
        "VCSELs and EML laser chips, silicon photonics, optical fibre, wavelength-division multiplexing"
    ),
    "Storage": (
        "Storing model weights, training datasets and KV inference caches — "
        "NAND flash, NVMe SSDs, high-bandwidth DRAM, persistent memory"
    ),
    "Semiconductor Manufacturing": (
        "Fabricating the chips — wafer fabs (foundries), lithography equipment, "
        "etch and deposition tools, advanced packaging, substrate manufacturers"
    ),
    "Energy Inputs": (
        "Upstream fuels and materials feeding power generation — natural gas production, "
        "uranium mining/enrichment, industrial gases (nitrogen, CO₂), dielectric cooling fluids"
    ),
    "Physical Infrastructure": (
        "Land, buildings and construction for data centres — "
        "site development, structural steel, raised flooring, fibre duct, cranes, fit-out contractors"
    ),
}

_SYSTEM = (
    "You are a contrarian physics arbitrage analyst who specialises in finding investment "
    "candidates the market has NOT yet connected to AI demand. You focus on under-the-radar "
    "companies where the AI discovery thesis is still early. You always return exactly the "
    "JSON format requested — no markdown, no prose, just the JSON array."
)

_PROMPT = """Find 6-8 publicly-traded companies that fit this physics arbitrage profile:

Strategy Type: {strategy_type}
{strategy_desc}

AI Stack Layer: {stack_layer}
{layer_desc}

Rules:
- Real companies with accurate, currently-traded tickers
- NOT widely known as AI plays — avoid NVDA, AMD, MSFT, GOOGL, AMZN, TSMC, ASML, INTC, QCOM
- Genuine physical constraint angle specific to this layer — not just "they sell to data centres"
- Prefer smaller or less-followed names where market awareness of the AI angle is below 30%
- Include global companies (Europe, Asia, Australia, Canada) where they are the best fit
- Skip any of these already-screened tickers: {existing_tickers}

Return ONLY a valid JSON array with no other text:
[
  {{
    "name": "Full Legal Company Name",
    "ticker": "TICKER",
    "exchange": "NYSE or NASDAQ or ASX or TSX or LSE etc",
    "rationale": "2 sentences: what specific physical thing does this company control or supply, and why does it become critical as AI scales?",
    "constraint": "The specific physical asset or component — 5 words or fewer",
    "market_awareness_pct": 12,
    "why_undiscovered": "One sentence: why hasn't the market made this AI connection yet?"
  }}
]"""


def generate_candidates(
    strategy_type: str,
    stack_layer: str,
    api_key: str,
    existing_tickers: list | None = None,
) -> list:
    """
    Ask Claude to generate candidate companies for a given strategy + stack layer.

    Args:
        strategy_type:    One of the keys in STRATEGY_DESCRIPTIONS
        stack_layer:      One of the keys in LAYER_DESCRIPTIONS
        api_key:          Anthropic API key
        existing_tickers: Tickers already in the screener (Claude will avoid them)

    Returns:
        List of candidate dicts with keys:
        name, ticker, exchange, rationale, constraint,
        market_awareness_pct, why_undiscovered
    """
    client = anthropic.Anthropic(api_key=api_key)

    existing_str = ", ".join(existing_tickers) if existing_tickers else "none"

    prompt = _PROMPT.format(
        strategy_type=strategy_type,
        strategy_desc=STRATEGY_DESCRIPTIONS.get(strategy_type, ""),
        stack_layer=stack_layer,
        layer_desc=LAYER_DESCRIPTIONS.get(stack_layer, ""),
        existing_tickers=existing_str,
    )

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=2048,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    return json.loads(match.group() if match else raw)
