"""
lstm_allocator.py - the LSTM that outputs portfolio weights directly.

Idea (after Zhang, Zohren & Roberts, 2020): skip the "forecast returns, then
optimise" step. The network reads 60 days of features and outputs weights.

Training is kept consistent with how the strategy trades:
  * one training sample per WEEKLY decision date (Friday close), not per day;
  * the label is the holding-period return actually earned by the backtest:
    from the execution close (next trading day) to the following week's
    execution close;
  * turnover is measured at the weekly rebalancing dates, against last week's
    weights after they have drifted with prices;
  * the loss is minus the annualised Sharpe ratio of the weekly returns net
    of cost (52 periods per year), plus lambda x mean weekly turnover.
So the turnover penalty tuned in training matches the costs paid in the backtest.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import config as C
from backtest import decision_dates


class LSTMAllocator(nn.Module):
    def __init__(self, n_features, n_assets, hidden=64, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, n_assets)

    def forward(self, x):
        h, _ = self.lstm(x)
        return capped_softmax(self.head(self.drop(h[:, -1])), C.W_MAX)


def capped_softmax(logits, cap, iters=10):
    """Long-only weights that sum to 1 with no weight above `cap`."""
    w = torch.softmax(logits, dim=-1)
    for _ in range(iters):
        excess = torch.relu(w - cap)
        if excess.sum() < 1e-9:
            break
        w = torch.minimum(w, torch.full_like(w, cap))
        free = (w < cap - 1e-9).float() * w
        w = w + excess.sum(-1, keepdim=True) * free / (free.sum(-1, keepdim=True) + 1e-12)
    return w


# ------------------------------------------------------------ weekly labels
def weekly_labels(prices, dec):
    """Holding-period return for each weekly decision date, exactly as the backtest earns it.

    Decision at close of d_k -> trade at close of e_k (next trading day) ->
    hold until the next trade at close of e_{k+1}.  Label_k = P(e_{k+1}) / P(e_k) - 1.
    Returns (labels DataFrame indexed by decision date, Series of label end dates).
    """
    idx = prices.index
    pos = idx.searchsorted(pd.DatetimeIndex(dec), side="right")      # execution positions
    rows, ends = {}, {}
    for k in range(len(dec) - 1):
        a, b = pos[k], pos[k + 1]
        if b >= len(idx):
            break
        rows[dec[k]] = prices.iloc[b].values / prices.iloc[a].values - 1
        ends[dec[k]] = idx[b]
    y = pd.DataFrame(rows, index=prices.columns).T
    return y, pd.Series(ends)


def make_windows(X, dates, y=None):
    """Stack LOOKBACK-day feature windows ending on each decision date."""
    Xv, pos = X.values.astype(np.float32), X.index
    xs, ys, ds = [], [], []
    for d in dates:
        i = pos.get_loc(d)
        if i + 1 < C.LOOKBACK:
            continue
        if y is not None and (d not in y.index or y.loc[d].isna().any()):
            continue
        xs.append(Xv[i + 1 - C.LOOKBACK:i + 1])
        if y is not None:
            ys.append(y.loc[d].values.astype(np.float32))
        ds.append(d)
    Y = torch.tensor(np.array(ys)) if y is not None else None
    return torch.tensor(np.array(xs)), Y, ds


# ------------------------------------------------------------ loss
def weekly_turnover(w, y):
    """Turnover at each weekly rebalance: |w_k - drift(w_{k-1})|, drift using last week's returns."""
    grown = w[:-1] * (1 + y[:-1])
    drifted = grown / grown.sum(-1, keepdim=True)
    return (w[1:] - drifted).abs().sum(-1)


def loss_fn(w, y, lam, cost=C.COST_BPS / 1e4, has_prev=False):
    """Minus annualised Sharpe of weekly net returns, plus lambda x mean weekly turnover.

    w, y are consecutive weekly samples.  If has_prev, row 0 is the previous week and is
    used only to measure the first turnover (it earns no return in this batch).
    """
    turn = weekly_turnover(w, y)
    if has_prev:
        w, y = w[1:], y[1:]
    else:                                   # no earlier week in this batch
        turn = torch.cat([torch.zeros(1), turn])
    r = (w * y).sum(-1) - cost * turn
    sharpe = r.mean() / (r.std() + 1e-8) * np.sqrt(C.PERIODS_PER_YEAR)
    return -sharpe + lam * turn.mean()


def train_one(Xtr, ytr, Xva, yva, cfg, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = LSTMAllocator(Xtr.shape[-1], ytr.shape[-1], cfg["hidden"], cfg["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=C.LR)
    best, best_state, wait = -np.inf, None, 0
    n, B = len(Xtr), C.BATCH_WEEKS
    for _ in range(C.MAX_EPOCHS):
        model.train()
        starts = np.random.permutation(np.arange(0, max(n - B, 1), B // 2))
        for s in starts:                    # contiguous blocks of weeks, shuffled order
            lo = max(s - 1, 0)              # include the previous week for the first turnover
            sl = slice(lo, s + B)
            opt.zero_grad()
            loss = loss_fn(model(Xtr[sl]), ytr[sl], cfg["lam"], has_prev=(lo < s))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = -loss_fn(model(Xva), yva, 0.0).item()   # validation: weekly net Sharpe
        if val > best + 1e-4:
            best, best_state, wait = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            wait += 1
            if wait >= C.PATIENCE:
                break
    model.load_state_dict(best_state)
    return model, best


def walk_forward(X, prices, fast=False, log=print):
    """Retrain each January on all earlier data; trade the year out of sample.
    Returns weekly LSTM weights and a log of every configuration tried (for the DSR)."""
    days = X.index
    prices = prices.loc[days]
    dec_all = list(pd.DatetimeIndex(decision_dates(days)))   # same weekly schedule as the backtest
    y, y_end = weekly_labels(prices, dec_all)
    seeds = 2 if fast else C.SEEDS
    grid = C.GRID[:2] if fast else C.GRID
    all_w, trials = {}, []
    for year in range(C.FIRST_TEST_YEAR, C.LAST_TEST_YEAR + 1):
        test_days = days[days.year == year]
        val_days = days[days.year == year - 1]
        if len(test_days) == 0 or len(val_days) == 0:
            continue
        # purge: a sample is usable only if its label ends PURGE_DAYS before the next block starts
        cut_tr = days[max(days.get_loc(val_days[0]) - C.PURGE_DAYS, 0)]
        cut_va = days[max(days.get_loc(test_days[0]) - C.PURGE_DAYS, 0)]
        tr_dec = [d for d in y.index if d < val_days[0] and y_end[d] <= cut_tr]
        va_dec = [d for d in y.index if val_days[0] <= d and y_end[d] <= cut_va]

        # scale using training data only
        tr_rows = days[days <= tr_dec[-1]]
        mu, sd = X.loc[tr_rows].mean(), X.loc[tr_rows].std().replace(0, 1)
        Xs = (X - mu) / sd
        Xtr, ytr, _ = make_windows(Xs, tr_dec, y)
        Xva, yva, _ = make_windows(Xs, va_dec, y)

        scores = []
        for cfg in grid:
            _, s = train_one(Xtr, ytr, Xva, yva, cfg, seed=0)
            scores.append(s)
            trials.append({"year": year, **cfg, "val_weekly_net_sharpe": s})
        best_cfg = grid[int(np.argmax(scores))]
        models = [train_one(Xtr, ytr, Xva, yva, best_cfg, seed=k)[0] for k in range(seeds)]
        log(f"{year}: {len(tr_dec)} train weeks, {len(va_dec)} val weeks; "
            f"best {best_cfg}; val weekly net Sharpe (ann.) {max(scores):.3f}")

        te_dec = [d for d in dec_all if d.year == year]
        Xte, _, te_dec = make_windows(Xs, te_dec)
        with torch.no_grad():
            w = torch.stack([m.eval()(Xte) for m in models]).mean(0).numpy()
        for d, row in zip(te_dec, w):
            all_w[d] = row
    W = pd.DataFrame(all_w, index=prices.columns).T
    return W, pd.DataFrame(trials)
