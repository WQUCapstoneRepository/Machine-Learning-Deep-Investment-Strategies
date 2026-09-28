"""
checks.py - mechanical checks run after `python run.py` (no study results are read here).

    python checks.py outputs/main        # or outputs/synthetic_main

Confirms that:
 1. every strategy's weights sum to one on every decision date, and capped rules never exceed 40%;
 2. trades are executed on the trading day after each decision;
 3. the LSTM's weekly training label equals the return the backtest earns between two trades;
 4. the LSTM's weekly training turnover equals the turnover the backtest charges.
"""
import sys, glob, os
import numpy as np
import pandas as pd
import config as C
import data as D
import backtest as BT
from lstm_allocator import weekly_labels

out = sys.argv[1] if len(sys.argv) > 1 else "outputs/synthetic_main"
synthetic = "synthetic" in out
prices = D.synthetic()[0] if synthetic else D.align(D.load_prices(), D.load_macro())[0]

ok = True
for f in sorted(glob.glob(os.path.join(out, "weights_*.csv"))):
    w = pd.read_csv(f, index_col=0, parse_dates=True)
    s_ok = np.allclose(w.sum(axis=1), 1, atol=1e-5)
    capped = not any(k in f for k in ("1-N", "60-40"))
    c_ok = (w.values.max() <= C.W_MAX + 1e-5) if capped else True
    print(f"{os.path.basename(f):28s} sum=1: {s_ok}   cap<=40%: {c_ok}   max={w.values.max():.3f}")
    ok &= s_ok and c_ok

# 2-4 on the LSTM (or HRP if the LSTM was not run)
name = "weights_LSTM.csv" if os.path.exists(os.path.join(out, "weights_LSTM.csv")) else "weights_HRP.csv"
W = pd.read_csv(os.path.join(out, name), index_col=0, parse_dates=True)
sim = BT.simulate(prices, W)
trade_days = sim.index[sim["turnover"] > 0]
nxt = [prices.index[i] for i in prices.index.searchsorted(W.index[1:], "right") if i < len(prices.index)]
t_ok = set(trade_days[1:]) <= set(nxt) | {trade_days[0]}
print("trades on the day after each decision:", t_ok); ok &= t_ok

dec = list(W.index)
y, y_end = weekly_labels(prices, dec)
gross = (1 + sim["gross"])
lab_ok, turn_ok = True, True
pos = prices.index.searchsorted(pd.DatetimeIndex(dec), "right")
for k in range(1, min(len(dec) - 2, 200)):
    a, b = prices.index[pos[k]], prices.index[pos[k + 1]]
    earned = gross.loc[a:b].iloc[1:].prod() - 1                   # backtest, days after trade a up to b
    label = float((W.iloc[k].values * y.iloc[k].values).sum())    # training label for week k
    lab_ok &= abs(earned - label) < 1e-8
    drifted = W.iloc[k - 1].values * (1 + y.iloc[k - 1].values); drifted /= drifted.sum()
    turn_ok &= abs(np.abs(W.iloc[k].values - drifted).sum() - sim.loc[a, "turnover"]) < 1e-8
print("weekly training label == backtest holding-period return:", lab_ok)
print("weekly training turnover == backtest turnover:", turn_ok)
ok &= lab_ok and turn_ok
print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
