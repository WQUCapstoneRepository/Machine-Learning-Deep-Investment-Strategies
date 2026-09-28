# Does a deep-learning allocator earn its keep?

MScFE 690 Capstone - Opolot Simon Peter & Uchenna Luke

We test whether an LSTM that outputs portfolio weights directly beats
Hierarchical Risk Parity (HRP) and four other classical rules on eight liquid
ETFs (SPY, EFA, EEM, TLT, LQD, GLD, DBC, VNQ), **after trading costs and out of
sample**, 2015-2025. We also ask three practical questions:

1. **Break-even cost** - at what trading cost does any LSTM advantage over HRP disappear?
2. **Cheap adaptivity check** - does the LSTM beat a simple 63-day inverse-volatility rule?
   If not, its extra complexity is not paying for itself.
3. **Stress windows** - how does each rule behave when diversification broke
   (Q4-2018, COVID crash 2020, 2022 stocks + bonds fall)?

Results are also reported separately for five sub-periods: 2015-2019 (pre-COVID),
2020 (COVID crash and recovery), 2021 (post-COVID recovery), 2022 (joint stock-bond
sell-off) and 2023-2025 (post-tightening). 2008 falls inside the initial training
window, so it is not part of the out-of-sample test.

## Files

| File | What it does |
|---|---|
| `config.py` | Every study setting in one place (universe, costs, walk-forward, grid) |
| `data.py` | Downloads and caches Yahoo Finance prices and FRED macro series; one-day macro lag |
| `features.py` | 48 price features + 3 macro features |
| `benchmarks.py` | HRP, 1/N, 60/40, min-variance and max-Sharpe (Ledoit-Wolf), inverse-volatility |
| `lstm_allocator.py` | LSTM with capped softmax output; trained on weekly holding-period returns with a weekly net-of-cost Sharpe loss and weekly turnover; walk-forward training |
| `backtest.py` | Daily backtester with drift and costs; metrics; break-even cost; LW (2008) test; Deflated Sharpe |
| `run.py` | Runs everything, writes tables and charts to `outputs/` |
| `checks.py` | Mechanical checks: weights sum to one, 40% cap, next-day execution, and training label / turnover identical to the backtest |

## Run

```bash
pip install -r requirements.txt
python run.py                         # benchmarks only (minutes) - initial results
python run.py --lstm                  # full study (CPU: under an hour)
python run.py --lstm --price-only     # RQ2: no macro inputs
python run.py --lstm --drop GLD       # RQ3: drop-one-asset run (repeat for each ticker)
python checks.py outputs/main         # mechanical checks on a finished run
python run.py --synthetic --lstm --fast   # smoke test on fake data
```

## Guardrails against fooling ourselves

* Decisions use data up to Friday close; trades happen at the next close.
* Training matches trading: one sample per weekly decision date, label = return from this
  week's trade to next week's trade, turnover measured at the weekly rebalancing dates
  (after drift), Sharpe annualised with 52 weeks. The cost the loss penalises is the cost
  the backtest charges.
* Macro series lagged one day; feature scaling fitted on training data only.
* 5-day purge: a training or validation label must end 5 trading days before the next block starts.
* Grid of 4 settings declared in advance; all trials logged for the Deflated Sharpe Ratio.
* Same cost, timing and 40% cap for every rule (1/N and 60/40 are fixed by definition).
