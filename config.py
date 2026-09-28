"""
config.py - one place for every study setting.

Changing a number here changes it everywhere, so the settings reported in the
paper and the settings used in the code can never drift apart.
"""

# ---------------------------------------------------------------- universe
TICKERS = ["SPY", "EFA", "EEM", "TLT", "LQD", "GLD", "DBC", "VNQ"]
START, END = "2007-01-01", "2025-12-31"

# FRED series: VIX, 10y-2y Treasury spread, broad US dollar index
MACRO = {"VIXCLS": "vix", "T10Y2Y": "curve", "DTWEXBGS": "usd"}

# ---------------------------------------------------------------- portfolio rules
W_MAX = 0.40            # no asset may exceed 40% of the portfolio
COST_BPS = 10           # proportional trading cost per unit of turnover (basis points)
COST_GRID_BPS = [0, 10, 25, 50]   # cost stress test
REBAL_FREQ = "W-FRI"    # decide on Friday close, trade on the next trading day's close
PERIODS_PER_YEAR = 52   # weekly rebalancing -> Sharpe in the training loss is annualised with 52
COV_WINDOW = 252        # look-back for HRP / min-var / max-Sharpe estimates (1 year)
INVVOL_WINDOW = 63      # look-back for the cheap adaptive rule (inverse volatility, ~3 months)

# ---------------------------------------------------------------- walk-forward
FIRST_TEST_YEAR, LAST_TEST_YEAR = 2015, 2025
PURGE_DAYS = 5          # a label must end 5 trading days before the next block (train/val/test) starts

# ---------------------------------------------------------------- LSTM
LOOKBACK = 60           # days of history the network sees
HORIZON = 5             # holding period: one week (5 trading days) = one rebalancing interval
# Training uses one sample per weekly decision date; the label is the return from this week's
# execution close to next week's, and turnover is measured at weekly rebalancing dates.
# lam multiplies mean WEEKLY turnover and is traded off against the ANNUALISED Sharpe.
GRID = [                # small, pre-declared grid -> counted in the Deflated Sharpe Ratio
    {"hidden": 32, "dropout": 0.2, "lam": 0.0},
    {"hidden": 64, "dropout": 0.2, "lam": 0.0},
    {"hidden": 32, "dropout": 0.2, "lam": 0.5},
    {"hidden": 64, "dropout": 0.2, "lam": 0.5},
]
SEEDS = 10              # final model = average of weights from 10 random seeds
MAX_EPOCHS = 100
PATIENCE = 10
BATCH_WEEKS = 26        # each mini-batch is a contiguous half-year of weekly samples
LR = 1e-3

# ---------------------------------------------------------------- stress windows (RQ3)
# Chosen in advance from well-known events, not from our results.
STRESS_WINDOWS = {
    "Q4-2018 sell-off":          ("2018-10-01", "2018-12-24"),
    "COVID crash 2020":          ("2020-02-19", "2020-03-23"),
    "2022 stocks+bonds fall":    ("2022-01-03", "2022-10-12"),
}
# Reporting sub-periods (each reported separately; 2020 and 2022 are the motivating episodes).
# 2008 lies inside the initial training window, so it is not part of the out-of-sample test.
SUB_PERIODS = {
    "2015-2019 (pre-COVID)":         ("2015-01-01", "2019-12-31"),
    "2020 (COVID crash & recovery)": ("2020-01-01", "2020-12-31"),
    "2021 (post-COVID recovery)":    ("2021-01-01", "2021-12-31"),
    "2022 (stock-bond sell-off)":    ("2022-01-01", "2022-12-31"),
    "2023-2025 (post-tightening)":   ("2023-01-01", "2025-12-31"),
}

DATA_DIR = "data"
OUT_DIR = "outputs"
