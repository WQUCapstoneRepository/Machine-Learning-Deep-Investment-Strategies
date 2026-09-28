"""
run.py - run the study end to end and write every table and chart to outputs/.

    python run.py                    # benchmarks only on real data (fast: initial results)
    python run.py --lstm             # + full LSTM walk-forward (all grid points, 10 seeds)
    python run.py --lstm --price-only    # RQ2 ablation: LSTM without macro inputs
    python run.py --lstm --drop GLD      # RQ3 robustness: drop one asset
    python run.py --synthetic --lstm --fast   # smoke test on FAKE data (checks the code runs)

Outputs: metrics.csv, table6.csv, cost_stress.csv, sub_periods.csv (Sharpe by period),
sub_periods_detail.csv (return, Sharpe, drawdown and LW test vs HRP for every period),
stress_windows.csv, significance.csv, lstm_trials.csv, weights_*.csv and PNG charts.
"""
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
import data as D
import features as F
import benchmarks as B
import backtest as BT

NAVY, GOLD, GREYS = "#1F3864", "#C9A227", ["#8C8C8C", "#A6A6A6", "#BFBFBF", "#6E6E6E", "#595959"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--lstm", action="store_true")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--price-only", action="store_true")
    ap.add_argument("--drop", default=None, help="ticker to leave out (drop-one-asset run)")
    a = ap.parse_args()

    tag = ("synthetic_" if a.synthetic else "") + ("price_only" if a.price_only else "main")
    if a.drop:
        tag += f"_drop_{a.drop}"
    out = os.path.join(C.OUT_DIR, tag)
    os.makedirs(out, exist_ok=True)

    # 1. data -------------------------------------------------------------
    if a.synthetic:
        prices, macro = D.synthetic()
        years = sorted(set(prices.index.year))
        C.FIRST_TEST_YEAR, C.LAST_TEST_YEAR = years[8], years[-1]
    else:
        prices, macro = D.align(D.load_prices(), D.load_macro())
    if a.drop:
        prices = prices.drop(columns=a.drop)
    daily = prices.pct_change().dropna()
    X = F.build_features(prices, macro, use_macro=not a.price_only)
    prices = prices.loc[X.index]

    test_start = f"{C.FIRST_TEST_YEAR}-01-01"
    dec = BT.decision_dates(prices.index, start=test_start)

    # 2. weights ----------------------------------------------------------
    weights = B.benchmark_weights(daily, dec)
    trials = None
    if a.lstm:
        import lstm_allocator as L
        weights["LSTM"], trials = L.walk_forward(X, prices, fast=a.fast)
        trials.to_csv(os.path.join(out, "lstm_trials.csv"), index=False)
    for k, w in weights.items():
        w.to_csv(os.path.join(out, f"weights_{k.replace('/', '-')}.csv"))

    # 3. simulate (one common start date so every strategy is judged on the same days)
    sims = {k: BT.simulate(prices, w) for k, w in weights.items() if len(w)}
    start = max(s.index[0] for s in sims.values())
    sims = {k: s.loc[start:] for k, s in sims.items()}
    nets = {k: BT.net(s) for k, s in sims.items()}

    # 4. tables -----------------------------------------------------------
    m = pd.DataFrame({k: BT.metrics(nets[k], sims[k]["turnover"]) for k in sims}).T
    m.round(4).to_csv(os.path.join(out, "metrics.csv"))

    cost = pd.DataFrame({c: {k: BT.metrics(BT.net(s, c))["Sharpe"] for k, s in sims.items()}
                         for c in C.COST_GRID_BPS})
    cost.columns = [f"Sharpe@{c}bps" for c in cost.columns]
    cost.round(3).to_csv(os.path.join(out, "cost_stress.csv"))

    # Table 6 layout used in the report
    t6 = pd.DataFrame({k: {"Ann. return": m.loc[k, "CAGR"], "Volatility": m.loc[k, "Vol"],
                           "Sharpe": m.loc[k, "Sharpe"], "Max. drawdown": m.loc[k, "MaxDD"],
                           "Turnover/year": m.loc[k, "Turnover/yr"],
                           "Return Jan-Oct 2022": BT.window_stats(nets[k], "2022-01-03", "2022-10-12")["Return"]
                           if len(nets[k].loc["2022-01-03":"2022-10-12"]) else np.nan}
                       for k in m.index}).T
    t6.round(4).to_csv(os.path.join(out, "table6.csv"))

    # sub-periods: each reported separately (2020 and 2022 on their own)
    periods = {p: (s, e) for p, (s, e) in C.SUB_PERIODS.items() if len(nets["HRP"].loc[s:e]) > 20}
    sub = pd.DataFrame({p: {k: BT.metrics(r.loc[s:e])["Sharpe"] for k, r in nets.items()}
                        for p, (s, e) in periods.items()})
    sub.round(3).to_csv(os.path.join(out, "sub_periods.csv"))
    rows = []
    for p, (s, e) in periods.items():
        for k, r in nets.items():
            x = r.loc[s:e]
            mm = BT.metrics(x, sims[k]["turnover"].loc[s:e])
            row = {"period": p, "strategy": k, "Return": (1 + x).prod() - 1,
                   "Ann. return": mm["CAGR"], "Sharpe": mm["Sharpe"], "MaxDD": mm["MaxDD"],
                   "Turnover/yr": mm["Turnover/yr"]}
            if "LSTM" in nets and k != "LSTM" and k == "HRP":
                row["LSTM-HRP Sharpe diff"], row["p-value (LW2008)"] = \
                    BT.sharpe_diff_test(nets["LSTM"].loc[s:e], x)
            rows.append(row)
    pd.DataFrame(rows).round(4).to_csv(os.path.join(out, "sub_periods_detail.csv"), index=False)

    rows = []
    for name, (s, e) in C.STRESS_WINDOWS.items():
        for k, r in nets.items():
            if len(r.loc[s:e]) > 3:
                rows.append({"window": name, "strategy": k, **BT.window_stats(r, s, e)})
    stress = pd.DataFrame(rows)
    if len(stress):
        stress.round(4).to_csv(os.path.join(out, "stress_windows.csv"), index=False)

    # 5. significance (LSTM vs each benchmark) ----------------------------
    if "LSTM" in sims:
        sig = []
        for k in sims:
            if k == "LSTM":
                continue
            d, p = BT.sharpe_diff_test(nets["LSTM"], nets[k])
            sig.append({"vs": k, "Sharpe diff (ann.)": d, "p-value (LW2008)": p,
                        "break-even cost bps": BT.break_even_cost(sims["LSTM"], sims[k])})
        n_trials = len(C.GRID) * C.SEEDS
        sig.append({"vs": f"Deflated Sharpe prob. (N={n_trials})",
                    "Sharpe diff (ann.)": np.nan,
                    "p-value (LW2008)": BT.deflated_sharpe(nets["LSTM"], n_trials),
                    "break-even cost bps": np.nan})
        pd.DataFrame(sig).round(4).to_csv(os.path.join(out, "significance.csv"), index=False)

    # 6. charts -----------------------------------------------------------
    charts(prices, nets, weights, out)
    print(m.round(3).to_string())
    print("\nSaved to", out)


def _style(k, i):
    if k == "LSTM":
        return dict(color=NAVY, lw=2.0, zorder=5)
    if k == "HRP":
        return dict(color=GOLD, lw=2.0, zorder=4)
    return dict(color=GREYS[i % len(GREYS)], lw=1.0, alpha=0.9)


def charts(prices, nets, weights, out):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, (k, r) in enumerate(nets.items()):
        ax.plot((1 + r).cumprod(), label=k, **_style(k, i))
    ax.set_title("Growth of $1, net of 10 bps costs")
    ax.grid(alpha=0.25); ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "cumulative.png"), dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    for i, (k, r) in enumerate(nets.items()):
        w = (1 + r).cumprod()
        ax.plot(w / w.cummax() - 1, label=k, **_style(k, i))
    ax.set_title("Drawdown from previous peak")
    ax.grid(alpha=0.25); ax.legend(frameon=False, ncol=4, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out, "drawdowns.png"), dpi=150); plt.close(fig)

    # the diversification regime our study is built around
    if {"SPY", "TLT"} <= set(prices.columns):
        lr = np.log(prices).diff()
        corr = lr["SPY"].rolling(252).corr(lr["TLT"]).dropna()
        fig, ax = plt.subplots(figsize=(9, 3))
        ax.plot(corr, color=NAVY, lw=1.5)
        ax.axhline(0, color="#595959", lw=0.8)
        ax.set_title("1-year rolling correlation, SPY vs TLT (stocks vs long bonds)")
        ax.grid(alpha=0.25)
        fig.tight_layout(); fig.savefig(os.path.join(out, "stock_bond_corr.png"), dpi=150); plt.close(fig)

    if "LSTM" in weights:
        w = weights["LSTM"]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.stackplot(w.index, w.T.values, labels=w.columns, edgecolor="white", linewidth=0.3)
        ax.set_ylim(0, 1); ax.set_title("LSTM weights through time")
        ax.legend(frameon=False, ncol=8, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.08))
        fig.tight_layout(); fig.savefig(os.path.join(out, "lstm_weights.png"), dpi=150); plt.close(fig)


if __name__ == "__main__":
    main()
