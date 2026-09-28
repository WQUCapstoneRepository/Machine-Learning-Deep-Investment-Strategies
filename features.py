"""
features.py - turn prices and macro series into model inputs.

Per asset (6 each x 8 assets = 48):
    log return over 1, 5, 21 days; rolling volatility over 21, 63 days; 126-day momentum
Macro (3): VIX level, yield-curve spread, 21-day change in the US dollar index
"""
import numpy as np
import pandas as pd


def build_features(prices, macro, use_macro=True):
    lr = np.log(prices).diff()
    blocks = {
        "r1": lr,
        "r5": lr.rolling(5).sum(),
        "r21": lr.rolling(21).sum(),
        "v21": lr.rolling(21).std() * np.sqrt(252),
        "v63": lr.rolling(63).std() * np.sqrt(252),
        "m126": lr.rolling(126).sum(),
    }
    parts = [b.add_prefix(f"{k}_") for k, b in blocks.items()]
    if use_macro:
        m = pd.DataFrame({
            "vix": macro["vix"],
            "curve": macro["curve"],
            "usd21": np.log(macro["usd"]).diff(21),
        })
        parts.append(m)
    X = pd.concat(parts, axis=1)
    return X.dropna()

# Training labels are weekly holding-period returns built in lstm_allocator.weekly_labels(),
# on the same weekly schedule the backtest trades, so they are not defined here.
