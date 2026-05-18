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
  "supply_response_score": <integer 1-10>,
  "category": "<exactly one of: Power Generation | Pre-Connected Power | Grid Infrastructure | Cooling and Power Management | Nuclear Revival | Critical Materials | Optical Interconnects | Other>",
  "subcategory": "<2-5 word description, e.g. Fuel Cells or Uranium Mining>",
  "cycle_stage": "<exactly one of: early | middle | late | mature>",
  "lead_time_advantage_yrs": <float, e.g. 5.0>,
  "constraint_controlled": "<one crisp sentence: what physical asset does this company control?>",
  "physics_thesis": "<2-3 sentences explaining why this company fits physics arbitrage>",
  "key_metrics": "<exactly 3 metrics to track, semicolon-separated>",
  "reasoning": "<2-3 sentences explaining how you arrived at the two scores>"
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

    response = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": EVALUATION_PROMPT.format(ticker=ticker, name=name, **context),
        }],
    )

    raw = response.content[0].text.strip()

    # Tolerate any leading/trailing prose around the JSON block
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    payload = json.loads(match.group() if match else raw)

    # Normalise keys and attach identifiers
    raw_cat = payload.get("category", "Other")
    category = raw_cat if raw_cat in VALID_CATEGORIES else "Other"

    return {
        "ticker":                  ticker.upper(),
        "name":                    name,
        "category":                category,
        "subcategory":             payload.get("subcategory", ""),
        "cycle_stage":             payload.get("cycle_stage", "early"),
        "lead_time_advantage_yrs": float(payload.get("lead_time_advantage_yrs", 3.0)),
        "ownership_score":         int(payload.get("ownership_score", 5)),
        "supply_response_score":   int(payload.get("supply_response_score", 5)),
        "constraint_controlled":   payload.get("constraint_controlled", ""),
        "physics_thesis":          payload.get("physics_thesis", ""),
        "key_metrics":             payload.get("key_metrics", ""),
        "reasoning":               payload.get("reasoning", ""),
        "yf_context":              context,                         # not saved to CSV
    }
