"""
backtest.py - a small, transparent daily backtester plus the performance measures.

How a trade works: weights decided at Friday close (date d) are traded at the
next trading day's close. Between trades, weights drift with prices. Cost on a
trade day = cost_rate x sum(|target weight - drifted weight|).

We keep gross returns and turnover separately, so net returns at ANY cost level
can be recomputed instantly (used for the cost stress test and break-even cost).
"""
import numpy as np
import pandas as pd
from scipy.stats import norm

import config as C


def decision_dates(index, start=None, end=None):
    s = pd.Series(index, index=index).loc[start:end]
    return list(s.groupby(s.index.to_period("W-FRI")).last().values)


def simulate(prices, target_w):
    """Return a DataFrame with daily gross return and turnover."""
    rets = prices.pct_change().fillna(0.0)
    idx = prices.index
    # shift each decision to the next trading day (execution day)
    exec_map = {}
    for d, w in target_w.iterrows():
        pos = idx.searchsorted(d, side="right")
        if pos < len(idx):
            exec_map[idx[pos]] = w.reindex(prices.columns).fillna(0).values
    start = min(exec_map)
    days = idx[idx >= start]
    w = np.zeros(prices.shape[1])
    gross, turn = [], []
    for day in days:
        r = rets.loc[day].values
        # today's return is earned on the weights held coming into today
        port_r = float(w @ r)
        w = w * (1 + r)
        w = w / w.sum() if w.sum() > 0 else w
        t = 0.0
        if day in exec_map:            # trade at today's close
            tgt = exec_map[day]
            t = float(np.abs(tgt - w).sum())
            w = tgt.copy()
        gross.append(port_r)
        turn.append(t)
    return pd.DataFrame({"gross": gross, "turnover": turn}, index=days)


def net(sim, cost_bps=C.COST_BPS):
    # cost of trading at today's close is charged against today's return
    return sim["gross"] - cost_bps / 1e4 * sim["turnover"]


# ------------------------------------------------------------ measures
def metrics(r, turnover=None):
    ann = 252
    wealth = (1 + r).cumprod()
    years = len(r) / ann
    cagr = wealth.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(ann)
    sharpe = r.mean() / r.std() * np.sqrt(ann) if r.std() > 0 else np.nan
    dd = wealth / wealth.cummax() - 1
    mdd = dd.min()
    q = r.quantile(0.05)
    cvar = r[r <= q].mean()
    out = {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": mdd,
           "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan, "CVaR95(daily)": cvar}
    if turnover is not None:
        out["Turnover/yr"] = turnover.sum() / years
    return out


def window_stats(r, start, end):
    x = r.loc[start:end]
    w = (1 + x).cumprod()
    return {"Return": w.iloc[-1] - 1, "MaxDD": (w / w.cummax() - 1).min()}


def break_even_cost(sim_a, sim_b, hi_bps=500):
    """Cost (bps) at which strategy A's net Sharpe falls to strategy B's.
    Returns None if A never beats B, or '>hi' if A still wins at hi_bps."""
    def gap(c):
        a, b = net(sim_a, c), net(sim_b, c)
        a, b = a.align(b, join="inner")
        return a.mean() / a.std() - b.mean() / b.std()
    if gap(0) <= 0:
        return None
    if gap(hi_bps) > 0:
        return f">{hi_bps}"
    lo, hi = 0.0, float(hi_bps)
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if gap(mid) > 0 else (lo, mid)
    return round(lo, 1)


# ------------------------------------------------------------ statistics
def sharpe_diff_test(ra, rb, lags=None):
    """Ledoit & Wolf (2008) HAC test of H0: Sharpe(A) = Sharpe(B). Returns (diff, p-value)."""
    ra, rb = ra.align(rb, join="inner")
    x = np.column_stack([ra, rb, ra ** 2, rb ** 2])
    T = len(x)
    m = x.mean(0)
    mu1, mu2, g1, g2 = m
    s1, s2 = np.sqrt(g1 - mu1 ** 2), np.sqrt(g2 - mu2 ** 2)
    diff = mu1 / s1 - mu2 / s2
    grad = np.array([g1 / s1 ** 3, -g2 / s2 ** 3, -mu1 / (2 * s1 ** 3), mu2 / (2 * s2 ** 3)])
    u = x - m
    lags = lags or int(4 * (T / 100) ** (2 / 9))
    psi = u.T @ u / T
    for L in range(1, lags + 1):          # Newey-West (Bartlett) weights
        g = u[L:].T @ u[:-L] / T
        psi += (1 - L / (lags + 1)) * (g + g.T)
    se = np.sqrt(grad @ psi @ grad / T)
    p = 2 * (1 - norm.cdf(abs(diff) / se))
    return diff * np.sqrt(252), p


def deflated_sharpe(r, n_trials, sr_trials_var=None):
    """Bailey & Lopez de Prado (2014). Probability the true Sharpe > 0 after
    allowing for the best of n_trials having been picked. Uses daily Sharpe."""
    sr = r.mean() / r.std()
    T = len(r)
    skew, kurt = r.skew(), r.kurt() + 3
    var_sr = sr_trials_var if sr_trials_var is not None else (1 / T)
    g = 0.5772156649
    sr0 = np.sqrt(var_sr) * ((1 - g) * norm.ppf(1 - 1 / n_trials)
                             + g * norm.ppf(1 - 1 / (n_trials * np.e)))
    z = (sr - sr0) * np.sqrt(T - 1) / np.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr ** 2)
    return float(norm.cdf(z))
