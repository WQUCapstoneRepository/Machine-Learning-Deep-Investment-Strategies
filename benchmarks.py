"""
benchmarks.py - the classical allocation rules the LSTM must beat.

Every function takes a window of daily returns that ends ON the decision date
(no future data) and returns long-only weights that sum to one.
"""
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

import config as C


def equal_weight(rets):
    n = rets.shape[1]
    return pd.Series(1.0 / n, rets.columns)


def sixty_forty(rets):
    w = pd.Series(0.0, rets.columns)
    w["SPY"], w["TLT"] = 0.6, 0.4
    return w


def inverse_vol(rets, window=C.INVVOL_WINDOW):
    """Cheap adaptive rule: hold less of whatever has recently been more volatile."""
    vol = rets.iloc[-window:].std()
    w = 1 / vol
    return cap_weights(w / w.sum())


# ------------------------------------------------------------ Hierarchical Risk Parity
def hrp(rets):
    """Lopez de Prado (2016): cluster by correlation, then split risk top-down."""
    cov, corr = rets.cov(), rets.corr()
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, None))
    link = linkage(squareform(dist.values, checks=False), method="single")
    order = list(rets.columns[leaves_list(link)])

    w = pd.Series(1.0, order)
    clusters = [order]
    while clusters:
        clusters = [c[j:k] for c in clusters
                    for j, k in ((0, len(c) // 2), (len(c) // 2, len(c))) if len(c) > 1]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            vl, vr = _cluster_var(cov, left), _cluster_var(cov, right)
            alpha = 1 - vl / (vl + vr)
            w[left] *= alpha
            w[right] *= 1 - alpha
    # same 40% cap as the LSTM, so we compare methods, not constraints
    return cap_weights(w[rets.columns])


def _cluster_var(cov, items):
    c = cov.loc[items, items].values
    ivp = 1 / np.diag(c)
    ivp /= ivp.sum()
    return float(ivp @ c @ ivp)


# ------------------------------------------------------------ Ledoit-Wolf optimisers
def _lw_cov(rets):
    return LedoitWolf().fit(rets.values).covariance_ * 252


def _solve(obj, n):
    cons = ({"type": "eq", "fun": lambda w: w.sum() - 1},)
    res = minimize(obj, np.full(n, 1 / n), bounds=[(0, C.W_MAX)] * n,
                   constraints=cons, method="SLSQP")
    w = np.clip(res.x, 0, None)
    return w / w.sum()


def min_variance(rets):
    S = _lw_cov(rets)
    return pd.Series(_solve(lambda w: w @ S @ w, S.shape[0]), rets.columns)


def max_sharpe(rets):
    S = _lw_cov(rets)
    mu = rets.mean().values * 252
    obj = lambda w: -(w @ mu) / np.sqrt(w @ S @ w + 1e-12)
    return pd.Series(_solve(obj, S.shape[0]), rets.columns)


# ------------------------------------------------------------ helpers
def cap_weights(w, cap=C.W_MAX, iters=20):
    """Clip any weight above the cap and hand the excess to the others pro-rata."""
    w = w.copy()
    for _ in range(iters):
        over = w > cap
        if not over.any():
            break
        excess = (w[over] - cap).sum()
        w[over] = cap
        under = ~over
        w[under] += excess * w[under] / w[under].sum()
    return w


RULES = {
    "HRP": (hrp, C.COV_WINDOW),
    "1/N": (equal_weight, 1),
    "60/40": (sixty_forty, 1),
    "MinVar-LW": (min_variance, C.COV_WINDOW),
    "MaxSharpe-LW": (max_sharpe, C.COV_WINDOW),
    "InvVol-63d": (inverse_vol, C.INVVOL_WINDOW),
}


def benchmark_weights(daily_rets, decision_dates):
    """Weights for every rule on every decision date, using data up to that date only."""
    out = {name: {} for name in RULES}
    for d in decision_dates:
        hist = daily_rets.loc[:d]
        for name, (fn, need) in RULES.items():
            if name == "60/40" and not {"SPY", "TLT"} <= set(daily_rets.columns):
                continue                    # 60/40 undefined when SPY or TLT is dropped
            if len(hist) >= need:
                out[name][d] = fn(hist.iloc[-max(need, 2):])
    return {k: pd.DataFrame(v).T for k, v in out.items() if v}
