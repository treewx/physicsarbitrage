import time
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf


def fetch_stock_data(tickers: List[str], period: str = "1y") -> Dict[str, dict]:
    """
    Fetch price + fundamental data for a list of tickers via yfinance.
    Returns a dict of ticker → data dict. Skips tickers that fail gracefully.
    """
    results: Dict[str, dict] = {}

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            info = t.info or {}
            hist = t.history(period=period)

            if hist.empty:
                results[ticker] = {"ticker": ticker, "error": "no history"}
                continue

            current_price = float(hist["Close"].iloc[-1])
            price_start = float(hist["Close"].iloc[0])
            perf_1y = (current_price / price_start - 1.0) * 100.0

            results[ticker] = {
                "ticker": ticker,
                "name": info.get("shortName", ticker),
                "current_price": current_price,
                "market_cap_b": (info.get("marketCap") or 0) / 1e9,
                "pe_ratio": info.get("forwardPE") or info.get("trailingPE"),
                "ev_ebitda": info.get("enterpriseToEbitda"),
                "revenue_growth": info.get("revenueGrowth"),
                "perf_1y": perf_1y,
                "sector": info.get("sector", "Unknown"),
                "history": hist["Close"],
            }
            time.sleep(0.15)

        except Exception as exc:
            results[ticker] = {"ticker": ticker, "error": str(exc)}

    return results


def fetch_price_history(tickers: List[str], period: str = "1y") -> pd.DataFrame:
    """
    Download closing prices for multiple tickers in one yfinance call.
    Returns a DataFrame with tickers as columns.
    """
    try:
        raw = yf.download(tickers, period=period, auto_adjust=True, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            return raw["Close"]
        return raw
    except Exception:
        return pd.DataFrame()


def market_data_to_df(market_data: Dict[str, dict]) -> pd.DataFrame:
    """Convert the dict returned by fetch_stock_data into a tidy display DataFrame."""
    rows = []
    for ticker, data in market_data.items():
        if "error" in data:
            continue
        pe = data.get("pe_ratio")
        ev = data.get("ev_ebitda")
        rg = data.get("revenue_growth")
        rows.append({
            "Ticker": ticker,
            "Price": f"${data['current_price']:.2f}",
            "Mkt Cap ($B)": f"{data['market_cap_b']:.1f}",
            "Fwd P/E": f"{pe:.1f}" if pe else "N/A",
            "EV/EBITDA": f"{ev:.1f}" if ev else "N/A",
            "Rev Growth": f"{rg * 100:.0f}%" if rg else "N/A",
            "1Y Perf": f"{data['perf_1y']:+.1f}%",
        })
    return pd.DataFrame(rows)
