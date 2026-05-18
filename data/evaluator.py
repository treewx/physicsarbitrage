"""
Claude-powered company evaluator for the Physics Arbitrage screener.

Given a ticker and company name, this module:
1. Fetches basic company context from Yahoo Finance
2. Sends it to the Claude API with a structured scoring rubric
3. Returns a dict that maps directly to a row in data/companies.csv
"""

import json
import re

import anthropic
import yfinance as yf

VALID_CATEGORIES = {
    "Power Generation", "Pre-Connected Power", "Grid Infrastructure",
    "Cooling and Power Management", "Nuclear Revival", "Critical Materials",
    "Optical Interconnects", "Other",
}


SYSTEM_PROMPT = (
    "You are an expert analyst evaluating public companies for a physics arbitrage "
    "investment screener based on Leopold Aschenbrenner's Situational Awareness framework. "
    "Physics arbitrage means finding companies that already CONTROL scarce physical "
    "infrastructure that major technology waves will need — before markets appreciate "
    "the scarcity. You are rigorous, honest, and willing to give low scores when a "
    "company has no genuine physics bottleneck angle."
)

EVALUATION_PROMPT = """Evaluate this company for a physics arbitrage screener focused on the AI / AGI scaling wave.

The physical bottleneck chain for AI is:
  AI training → GPU clusters → Electric power → Grid connection → Power generation
  → Cooling systems → Transformers & copper wiring → Mining (copper, uranium)

The highest-scoring companies in this framework are ones like:
  • Bloom Energy (BE) — fuel cells that bypass the 5-7yr grid connection queue
  • Core Scientific (CORZ) — Bitcoin miner with pre-connected power converting to AI hosting
  • Vertiv (VRT) — only liquid cooling solution for AI chips that physically cannot be air-cooled
  • Cameco (CCJ) — uranium miner; mine development takes 10-20 years; irreplaceable

Company to evaluate:
  Ticker:     {ticker}
  Name:       {name}
  Sector:     {sector}
  Industry:   {industry}
  Market Cap: ${market_cap_b}B
  Business:   {description}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SCORING RUBRIC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ownership_score (1-10) — Does this company OWN or CONTROL a physical bottleneck asset?
  10  = Owns a geologically / physically scarce asset with NO substitutes
        (Examples: uranium mine, pre-connected power site, nuclear plant, sole liquid-cooling OEM)
  8-9 = Primary controller of a scarce physical asset; very limited competition;
        5+ years for anyone to replicate from scratch
        (Examples: Cameco dominates Western uranium supply; Vertiv is the dominant liquid-cooling supplier)
  6-7 = Directly OWNS OR OPERATES physical infrastructure, but 2-5 real competitors exist;
        2-5 year replication time. ONLY use this band if the company physically operates the
        asset — NOT if it merely designs, sells equipment into, or services the bottleneck.
  4-5 = Exposed to the bottleneck theme but does NOT own the physical asset.
        This includes: fabless chip designers (NVIDIA designs GPUs but TSMC manufactures them → 4),
        equipment vendors that sell into the bottleneck but don't control it, software/cloud layers,
        companies with indirect exposure through a diversified business.
  1-3 = Tangential, indirect, or no genuine physics link.
        Holding companies with no direct operations, pure software businesses, diversified
        conglomerates where the bottleneck is a minor segment.

⚠ CALIBRATION — give low scores freely. Market cap, brand recognition, and revenue are
IRRELEVANT to this score. Only direct physical ownership counts. NVIDIA (fabless, $3T market cap)
scores LOWER than an obscure uranium miner that owns proven in-ground reserves. If torn between
two bands, choose the LOWER one. The rubric rewards physical control, not financial size.

supplier_score (1-10) — Is this company a MANDATORY SUPPLIER of a critical bottleneck component?
  10  = Sole or dominant global supplier of a physical component without which the bottleneck
        cannot function at scale. No viable substitute exists.
        (Examples: Lumentum — dominant supplier of laser chips for AI optical transceivers;
         ASML — only maker of EUV machines)
  8-9 = One of 2-3 globally qualified suppliers; switching requires years of qualification;
        no new entrant can be production-ready in under 5 years
  6-7 = Important supplier but 3-5 qualified competitors exist; can be substituted in 2-4 years
  4-5 = Sells into the bottleneck but the product is relatively commoditised or substitutable
  1-3 = Peripheral vendor, easily switched, no unique technical capability tied to the bottleneck
  1   = Company is primarily an asset owner, not a component manufacturer — give it 1 here
        and score it on ownership_score instead

  NOTE: Score each dimension independently. Most companies score high on ONE of
  ownership_score or supplier_score, and low on the other. That is expected.

supply_response_score (1-10) — How fast can new supply come online, even with unlimited capital?
  10  = CANNOT be meaningfully accelerated (copper mine = 10-20 years; nuclear plant = 10 years)
  8-9 = 5-10 years even with full capital commitment
  6-7 = 2-5 years
  4-5 = 1-2 years
  2-3 = 6-12 months
  1   = Commodity market — responds within weeks / months regardless of price signal

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
OUTPUT FORMAT
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Respond ONLY with a valid JSON object and absolutely no other text:

{{
  "ownership_score": <integer 1-10>,
  "supplier_score": <integer 1-10>,
  "supply_response_score": <integer 1-10>,
  "strategy_type": "<one or more comma-separated from: Asset Control | Mandatory Supplier | Conversion Optionality | Input Supply | Special Situation>",
  "category": "<exactly one of: Power Generation | Pre-Connected Power | Grid Infrastructure | Cooling and Power Management | Nuclear Revival | Critical Materials | Optical Interconnects | Other>",
  "subcategory": "<2-5 word description, e.g. Fuel Cells or Uranium Mining>",
  "cycle_stage": "<exactly one of: early | middle | late | mature>",
  "lead_time_advantage_yrs": <float, e.g. 5.0>,
  "constraint_controlled": "<one crisp sentence: what physical asset does this company control?>",
  "physics_thesis": "<2-3 sentences explaining why this company fits physics arbitrage>",
  "key_metrics": "<exactly 3 metrics to track, semicolon-separated>",
  "reasoning": "<2-3 sentences explaining how you arrived at the ownership, supplier, and supply_response scores>"
}}

If this company has no meaningful physics arbitrage angle give it ownership_score 1-3 and be explicit about why in reasoning.
A fabless semiconductor designer, a pure-software AI company, or a holding company with no direct
physical operations should score 1-4 on ownership_score. Do not let name recognition or market cap
push the score up — score the physics, not the brand."""


def _fetch_context(ticker: str) -> dict:
    """Pull basic company info from Yahoo Finance."""
    try:
        info = yf.Ticker(ticker).info or {}
        desc = (info.get("longBusinessSummary") or "").strip()
        return {
            "sector":       info.get("sector", "Unknown"),
            "industry":     info.get("industry", "Unknown"),
            "description":  desc[:700] if desc else "No description available.",
            "market_cap_b": round((info.get("marketCap") or 0) / 1e9, 1),
        }
    except Exception:
        return {
            "sector": "Unknown", "industry": "Unknown",
            "description": "Could not fetch company data.", "market_cap_b": 0,
        }


def evaluate_company(ticker: str, name: str, api_key: str) -> dict:
    """
    Ask Claude to evaluate a company and return a dict ready to append to companies.csv.

    Raises:
        json.JSONDecodeError  — if Claude's response is not valid JSON
        anthropic.APIError    — if the API call fails
    """
    context = _fetch_context(ticker)

    client = anthropic.Anthropic(api_key=api_key)

    messages = [{
        "role": "user",
        "content": EVALUATION_PROMPT.format(ticker=ticker, name=name, **context),
    }]

    payload = None
    for _attempt in range(3):
        response = client.messages.create(
            model="claude-opus-4-7",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=messages,
        )

        raw = response.content[0].text.strip()

        # Tolerate any leading/trailing prose around the JSON block
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        candidate = match.group() if match else raw

        try:
            payload = json.loads(candidate)
            break
        except json.JSONDecodeError as exc:
            if _attempt == 2:
                raise
            messages = messages + [
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        f"Your previous response contained invalid JSON "
                        f"(error: {exc}). "
                        "Output ONLY the corrected JSON object — "
                        "no prose, no markdown fences. "
                        "Escape any apostrophes inside string values as \\u0027."
                    ),
                },
            ]

    # Normalise keys and attach identifiers
    raw_cat = payload.get("category", "Other")
    category = raw_cat if raw_cat in VALID_CATEGORIES else "Other"

    valid_strategies = {"Asset Control", "Mandatory Supplier", "Conversion Optionality",
                        "Input Supply", "Special Situation"}
    raw_strat = payload.get("strategy_type", "Asset Control")
    # Keep only recognised tokens
    strategy_type = ", ".join(
        s.strip() for s in raw_strat.split(",") if s.strip() in valid_strategies
    ) or "Asset Control"

    return {
        "ticker":                  ticker.upper(),
        "name":                    name,
        "category":                category,
        "subcategory":             payload.get("subcategory", ""),
        "strategy_type":           strategy_type,
        "cycle_stage":             payload.get("cycle_stage", "early"),
        "lead_time_advantage_yrs": float(payload.get("lead_time_advantage_yrs", 3.0)),
        "ownership_score":         int(payload.get("ownership_score", 5)),
        "supplier_score":          int(payload.get("supplier_score", 1)),
        "supply_response_score":   int(payload.get("supply_response_score", 5)),
        "constraint_controlled":   payload.get("constraint_controlled", ""),
        "physics_thesis":          payload.get("physics_thesis", ""),
        "key_metrics":             payload.get("key_metrics", ""),
        "reasoning":               payload.get("reasoning", ""),
        "yf_context":              context,                         # not saved to CSV
    }
