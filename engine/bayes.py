"""
Stage six: do the conditions compose, and does that survive out of sample?

Every stage before this one measures conditions *one at a time*. The obvious next
question -- what happens when several hold at once -- has an answer already sitting in
the artifacts. Under conditional independence the log-odds add:

    logit P(touch theta in h | x1..xk)
        = logit P(touch theta in h) + sum_i [ logit P(touch theta in h | xi)
                                              - logit P(touch theta in h) ]

Every term on the right is a cube cell and the prior is the baseline node, so a naive
Bayes ensemble over this universe is a *sum over existing measurements*: no new
estimator, no fitted weights, nothing but the table the pipeline already built.

Two things make that honest rather than a demo.

**It is fit out of sample.** Every number the pipeline publishes elsewhere is measured
over the full history, which is correct for describing what happened and useless for
claiming prediction. Here the bin edges, the per-bin rates and the prior are estimated
on a training window only, then applied unchanged to a later window the fit never saw.
The walk forward is expanding: fit on everything up to a point, predict the block after
it, roll forward, repeat.

**The overlap is purged.** A label at bar t looks h bars into the future, so a training
label within h bars of the test block has already seen part of it. Those bars are
dropped -- the embargo -- and the test block is scored on non-overlapping bars, every
h-th one, because 400 overlapping windows are not 400 observations and a calibration
curve drawn on them claims a precision the sample cannot support.

The independence assumption is false here and visibly so: `ma_ratio_7`, `roc_5` and
`macd_12_26` are near-restatements of each other, and adding all three counts one piece
of evidence three times. That failure is the deliverable rather than something to hide
-- `--report` draws the all-nodes model and the one-per-family model on the same
calibration axes, and the gap between them is the cost of the assumption, measured.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from engine.barrier import forward_extremes_upto

# Shrinkage strength for a bin's rate, in pseudo-observations drawn from the prior.
# A decile bin on a training window holds a few hundred bars, so k = 25 moves a
# well-populated bin barely at all while keeping a thin one from contributing a
# runaway log-odds term to the sum.
SHRINK_K = 25.0

# Probabilities are clipped this far from 0 and 1 before the logit, so one saturated
# term cannot dominate.
_EPS = 1e-6


def touch_label(data: pd.DataFrame, theta: float, h: int) -> np.ndarray:
    """
    1 where price touches theta within h bars of t, 0 where it does not, NaN where the
    forward window is not realized.

    Uses the same intraday high/low excursions as the cube, so the event predicted here
    and the event measured everywhere else are the same event.
    """
    mins, maxs = forward_extremes_upto(data, h)
    excursion = mins[h - 1] if theta < 0 else maxs[h - 1]
    y = (excursion <= theta) if theta < 0 else (excursion >= theta)
    return np.where(np.isnan(excursion), np.nan, y.astype(float))


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p))


def fit(y: np.ndarray, bins: np.ndarray, n_bins: np.ndarray,
        k: float = SHRINK_K) -> dict:
    """
    Per-feature log-odds contributions, estimated on one training window.

    `bins[i, t]` is feature i's bin index at bar t and `n_bins[i]` its bin count.
    Returns the prior log-odds and a ragged list of per-bin *deltas* -- each already the
    `logit P(event | bin) - logit P(event)` term the sum adds, so prediction is a gather
    and an add with no arithmetic left in it.

    Each bin's rate is shrunk toward the prior by k pseudo-observations. A bin holding
    300 bars barely moves; one holding four collapses to the prior instead of
    contributing a spurious swing to the score.
    """
    prior = float(np.mean(y))
    lp = _logit(prior)

    deltas = []
    for i, nb in enumerate(n_bins):
        hits = np.bincount(bins[i], weights=y, minlength=int(nb))[:int(nb)]
        n = np.bincount(bins[i], minlength=int(nb))[:int(nb)].astype(float)
        rate = (hits + k * prior) / (n + k)
        deltas.append(_logit(rate) - lp)
    return {'prior': prior, 'logit_prior': float(lp), 'deltas': deltas}


def score(model: dict, bins: np.ndarray, use: np.ndarray | None = None) -> np.ndarray:
    """
    Summed log-odds per bar for the features in `use`.

    `use` is a boolean mask over features; omitted, every feature contributes. That mask
    is the only difference between the all-nodes model and the one-per-family model --
    same table, different subset of terms.
    """
    n_feat, n_t = bins.shape
    idx = range(n_feat) if use is None else np.flatnonzero(use)
    s = np.full(n_t, model['logit_prior'])
    for i in idx:
        s = s + model['deltas'][i][bins[i]]
    return s


def predict(model: dict, bins: np.ndarray, use: np.ndarray | None = None) -> np.ndarray:
    """P(event) per bar: the summed log-odds, squashed."""
    return 1.0 / (1.0 + np.exp(-score(model, bins, use)))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Overflow-free logistic: exp is only ever applied to a non-positive argument."""
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    e = np.exp(z[~pos])
    out[~pos] = e / (1.0 + e)
    return out


def platt(s: np.ndarray, y: np.ndarray, iters: int = 100) -> tuple[float, float]:
    """
    Fit `P(y=1) = sigmoid(a*s + b)`: the one-dimensional rescaling that turns a good
    ranking into an honest probability.

    Naive Bayes over correlated features double-counts evidence, so its scores come out
    stretched -- the ordering survives that, the scale does not. Two parameters put the
    scale back without touching the ordering, which is why AUC is unchanged by this step
    and only the Brier score moves. `a` < 1 is the measured overconfidence.

    Two details keep the fit from running away, both of them Platt's (1999, §2.1):

    **Smoothed targets.** Maximum likelihood against hard 0/1 labels is unbounded
    whenever the training scores separate -- the likelihood keeps rising as a goes to
    infinity. Fitting against (N+ + 1)/(N+ + 2) and 1/(N- + 2) instead is a proper prior
    on the two classes and makes the optimum finite.

    **Damped Newton.** The raw naive-Bayes score saturates the logistic at |s| ~ 20, so
    the IRLS weights p(1-p) underflow and the Hessian goes singular; an undamped step
    then jumps to a ~ 1e8. A ridge on the Hessian plus backtracking on the objective
    keeps every step a descent step.

    Fit on the training window and applied unchanged to the test block, like every other
    quantity here.
    """
    s = np.asarray(s, dtype=float)
    n_pos = float(np.sum(y == 1))
    n_neg = float(np.sum(y == 0))
    hi = (n_pos + 1.0) / (n_pos + 2.0)
    lo = 1.0 / (n_neg + 2.0)
    t = np.where(y == 1, hi, lo)

    def nll(a: float, b: float) -> float:
        z = a * s + b
        # log(1 + exp(z)) evaluated stably
        soft = np.maximum(z, 0.0) + np.log1p(np.exp(-np.abs(z)))
        return float(np.sum(soft - t * z))

    a, b = 0.0, float(_logit(np.mean(t)))
    cur = nll(a, b)
    for _ in range(iters):
        p = _sigmoid(a * s + b)
        w = p * (1.0 - p)
        r = t - p
        g = np.array([np.dot(r, s), r.sum()])
        H = np.array([[np.dot(w * s, s), np.dot(w, s)],
                      [np.dot(w, s),     w.sum()]])
        H[0, 0] += 1e-10
        H[1, 1] += 1e-10
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break

        # backtrack until the step actually decreases the objective
        scale = 1.0
        for _ in range(40):
            cand = nll(a + scale * step[0], b + scale * step[1])
            if cand <= cur:
                break
            scale *= 0.5
        else:
            break
        a, b = a + scale * step[0], b + scale * step[1]
        if abs(cur - cand) < 1e-12:
            cur = cand
            break
        cur = cand
    return float(a), float(b)


# -- scoring -------------------------------------------------------------------

def brier(y: np.ndarray, p: np.ndarray) -> float:
    """
    Mean squared error of a probability forecast; lower is better.

    The prior-only model scores p(1-p), and that is the number every other model has to
    beat before any of its structure counts for anything.
    """
    return float(np.mean((p - y) ** 2))


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """
    Area under the ROC curve by the rank identity (Mann-Whitney U), ties averaged.

    Ranking is the one claim naive Bayes makes credibly even when its probabilities are
    miscalibrated, so this and the calibration curve answer different questions and both
    are reported.
    """
    pos, neg = y == 1, y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float('nan')

    order = np.argsort(p, kind='stable')
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)

    s = p[order]
    start = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            ranks[order[start:i]] = ranks[order[start:i]].mean()
            start = i
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def calibration(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """
    Predicted against realized frequency, over equal-count bins of the forecast.

    Equal-count rather than equal-width: a model that never predicts above 40% leaves
    the top half of an equal-width axis empty, and the curve then says more about the
    binning than about the forecast.
    """
    if len(p) < n_bins * 2:
        return {'predicted': np.array([]), 'realized': np.array([]),
                'count': np.array([])}
    order = np.argsort(p, kind='stable')
    groups = np.array_split(order, n_bins)
    return {
        'predicted': np.array([p[g].mean() for g in groups]),
        'realized':  np.array([y[g].mean() for g in groups]),
        'count':     np.array([len(g) for g in groups]),
    }


# -- the walk forward ----------------------------------------------------------

def walk_forward(
    data:       pd.DataFrame,
    feats:      dict,
    families:   dict,
    theta:      float,
    horizon:    int,
    n_bins:     int = 10,
    folds:      int = 5,
    start_frac: float = 0.5,
) -> dict:
    """
    Expanding-window walk forward over `folds` test blocks covering the last
    (1 - start_frac) of the sample.

    Per fold, in this order:

      1. cut the training window at the fold's start, then drop its last `horizon`
         bars -- the embargo -- because their labels reach into the test block
      2. estimate bin edges, per-bin rates and the prior on what is left
      3. rank features by training-window separation and keep one per family, so the
         deduplication is itself fit out of sample rather than chosen with hindsight
         over the whole history
      4. predict the test block with the edges and tables from step 2
      5. score every `horizon`-th test bar, so the scored windows do not overlap

    Returns the pooled out-of-sample predictions of both models, the prior-only
    reference, and the per-fold boundaries.
    """
    y_all = touch_label(data, theta, horizon)
    names = list(feats)
    fam = [families[n] for n in names]
    X = np.vstack([feats[n].to_numpy(float) for n in names])

    ok = np.isfinite(y_all) & np.isfinite(X).all(axis=0)
    y_all, X = y_all[ok], X[:, ok]
    idx_all = np.flatnonzero(ok)
    n = len(y_all)

    cut = int(n * start_frac)
    bounds = np.linspace(cut, n, folds + 1).astype(int)
    cuts = np.linspace(0, 1, n_bins + 1)[1:-1]

    out = {k: [] for k in ('y', 'p_all', 'p_dedup', 'p_scaled', 'p_prior', 'fold')}
    fold_rows, kept_per_fold = [], []

    for f in range(folds):
        t0, t1 = int(bounds[f]), int(bounds[f + 1])
        train_end = t0 - horizon                      # the embargo
        if train_end < 200 or t1 - t0 < horizon * 2:
            continue

        y_tr, X_tr = y_all[:train_end], X[:, :train_end]

        # edges from the training window only, so a test bar is binned by what was
        # knowable at the time and never by its own future distribution
        edges = [np.unique(np.quantile(X_tr[i], cuts)) for i in range(len(names))]
        nb = np.array([len(e) + 1 for e in edges])
        b_tr = np.vstack([np.searchsorted(edges[i], X_tr[i]) for i in range(len(names))])

        model = fit(y_tr, b_tr, nb)

        # one feature per family: the largest training-window spread in log-odds, the
        # same "does this condition move the rate" question the economic filter asks,
        # restricted to data this fold is allowed to see
        spread = np.array([float(np.ptp(d)) for d in model['deltas']])
        keep = np.zeros(len(names), dtype=bool)
        for family in sorted(set(fam)):
            members = [i for i in range(len(names)) if fam[i] == family]
            keep[max(members, key=lambda i: spread[i])] = True
        kept_per_fold.append(sorted(names[i] for i in np.flatnonzero(keep)))

        # the scale correction is fit on the training window too, so it sees no more
        # than the tables it is correcting
        a, b = platt(score(model, b_tr, keep), y_tr)

        sl = slice(t0, t1)
        b_te = np.vstack([np.searchsorted(edges[i], X[i, sl]) for i in range(len(names))])

        # non-overlapping scoring bars: consecutive windows share h-1 bars of future
        step = np.zeros(t1 - t0, dtype=bool)
        step[::horizon] = True

        s_te = score(model, b_te, keep)
        out['y'].append(y_all[sl][step])
        out['p_all'].append(predict(model, b_te)[step])
        out['p_dedup'].append((1.0 / (1.0 + np.exp(-s_te)))[step])
        out['p_scaled'].append((1.0 / (1.0 + np.exp(-(a * s_te + b))))[step])
        out['p_prior'].append(np.full(int(step.sum()), model['prior']))
        out['fold'].append(np.full(int(step.sum()), f))

        fold_rows.append({
            'fold': f,
            'train_bars': int(train_end),
            'test_bars': int(t1 - t0),
            'scored_bars': int(step.sum()),
            'train_end':  str(data.index[idx_all[train_end - 1]])[:10],
            'test_start': str(data.index[idx_all[t0]])[:10],
            'test_end':   str(data.index[idx_all[t1 - 1]])[:10],
            'prior': model['prior'],
            'n_kept': int(keep.sum()),
            'platt_a': a, 'platt_b': b,
        })

    if not fold_rows:
        raise ValueError('not enough history to walk forward at this horizon')

    res = {k: np.concatenate(v) for k, v in out.items()}
    res.update({'folds': fold_rows, 'kept': kept_per_fold,
                'theta': float(theta), 'horizon': int(horizon),
                'nodes': names, 'n_features': len(names)})
    # The prior-only forecast is constant within a fold, so it ranks nothing and its AUC
    # is undefined. Pooling the folds would give it one anyway -- a number driven purely
    # by the prior drifting between folds, which reads as discrimination and is not.
    res['metrics'] = {
        name: {'brier': brier(res['y'], res[key]),
               'auc': float('nan') if key == 'p_prior' else auc(res['y'], res[key]),
               'mean_predicted': float(res[key].mean())}
        for name, key in (('all_nodes', 'p_all'),
                          ('one_per_family', 'p_dedup'),
                          ('one_per_family_scaled', 'p_scaled'),
                          ('prior_only', 'p_prior'))
    }
    res['metrics']['realized_rate'] = float(res['y'].mean())
    res['metrics']['n_scored'] = int(len(res['y']))
    return res
