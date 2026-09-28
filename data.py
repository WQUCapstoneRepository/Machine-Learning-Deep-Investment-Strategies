"""
data.py - download, cache and align prices (Yahoo Finance) and macro series (FRED).

Rule: every macro value is lagged one day, so a decision on day t only uses
numbers that were actually published by day t.
"""
import os
import io
import urllib.request
import numpy as np
import pandas as pd

import config as C


def _cache(name):
    os.makedirs(C.DATA_DIR, exist_ok=True)
    return os.path.join(C.DATA_DIR, name)


def load_prices(tickers=C.TICKERS, start=C.START, end=C.END, refresh=False):
    path = _cache("prices.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)[tickers]
    import yfinance as yf
    px = yf.download(tickers, start=start, end=end, auto_adjust=True,
                     progress=False)["Close"][tickers]
    px = px.dropna(how="all").ffill().dropna()
    px.to_csv(path)
    return px


def load_macro(start=C.START, end=C.END, refresh=False):
    path = _cache("macro.csv")
    if os.path.exists(path) and not refresh:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    frames = []
    for code, name in C.MACRO.items():
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={code}"
        raw = urllib.request.urlopen(url, timeout=30).read().decode()
        s = pd.read_csv(io.StringIO(raw), index_col=0, parse_dates=True,
                        na_values=".").iloc[:, 0].rename(name)
        frames.append(s)
    m = pd.concat(frames, axis=1).loc[start:end]
    m.to_csv(path)
    return m


def align(prices, macro):
    """Put macro on the trading calendar, forward-fill gaps, lag one day."""
    macro = macro.reindex(prices.index).ffill().shift(1)
    ok = macro.notna().all(axis=1)
    return prices[ok], macro[ok]


def synthetic(n_days=4750, seed=0):
    """Fake data with calm and stressed regimes. ONLY for testing that the code runs."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2007-01-03", periods=n_days)
    k = len(C.TICKERS)
    regime = (np.sin(np.arange(n_days) / 400) > 0.6).astype(int)
    mu = np.array([[4, 4, 5, 2, 2, 2, 1, 4], [-20, -22, -28, 8, -4, 6, -15, -25]]) / 1e4
    vol = np.array([[1.0, 1.1, 1.4, 0.8, 0.5, 0.9, 1.1, 1.3],
                    [3.0, 3.2, 3.8, 1.2, 1.0, 1.5, 2.5, 3.5]]) / 100
    base_corr = 0.3 + 0.7 * np.eye(k)
    L = np.linalg.cholesky(base_corr)
    z = rng.standard_normal((n_days, k)) @ L.T
    r = mu[regime] + vol[regime] * z
    prices = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), idx, C.TICKERS)
    macro = pd.DataFrame({
        "vix": 15 + 20 * regime + rng.normal(0, 2, n_days),
        "curve": 1 + np.cumsum(rng.normal(0, 0.02, n_days)),
        "usd": 100 * np.exp(np.cumsum(rng.normal(0, 0.003, n_days))),
    }, idx)
    return prices, macro
