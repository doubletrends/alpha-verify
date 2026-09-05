"""
Stage six: do the conditions compose, and does that survive out of sample?

Every stage before this one measures conditions *one at a time*. The obvious next
question -- what happens when several hold at once -- has an answer already sitting in
the artifacts. Under conditional independence the log-odds add:

    logit P(touch Δ in t | x1..xk)
        = logit P(touch Δ in t) + sum_i [ logit P(touch Δ in t | xi)
                                              - logit P(touch Δ in t) ]

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

**The overlap is purged.** A label at bar t looks t bars into the future, so a training
label within t bars of the test block has already seen part of it. Those bars are
dropped -- the embargo -- and the test block is scored on non-overlapping bars, every
t-th one, because 400 overlapping windows are not 400 observations and a calibration
curve drawn on them claims a precision the sample cannot support.

The independence assumption is false here and visibly so: related momentum and
volatility nodes can count one piece of evidence several times. The weighted model
cross-fits each node's log-odds contribution inside the training window, then learns
non-negative ridge-logistic reliability weights. Ridge shares evidence across correlated
nodes rather than forcing one representative to erase the others' unique information.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from barrierlab.domain.barrier import forward_extremes_upto
from barrierlab.domain import redundancy

# Shrinkage strength for a bin's rate, in pseudo-observations drawn from the prior.
# A decile bin on a training window holds a few hundred bars, so k = 25 moves a
# well-populated bin barely at all while keeping a thin one from contributing a
# runaway log-odds term to the sum.
SHRINK_K = 25.0

# Probabilities are clipped this far from 0 and 1 before the logit, so one saturated
# term cannot dominate.
_EPS = 1e-6


def touch_label(data: pd.DataFrame, Δ: float, t: int) -> np.ndarray:
    """
    1 where price touches Δ within t bars of t, 0 where it does not, NaN where the
    forward window is not realized.

    Uses the same intraday high/low excursions as the cube, so the event predicted here
    and the event measured everywhere else are the same event.
    """
    mins, maxs = forward_extremes_upto(data, t)
    excursion = mins[t - 1] if Δ < 0 else maxs[t - 1]
    y = (excursion <= Δ) if Δ < 0 else (excursion >= Δ)
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


def information_scores(model: dict, bins: np.ndarray, n_bins: np.ndarray) -> np.ndarray:
    """Expected log-loss gain of each complete bin table over the prior, in nats."""
    prior = float(np.clip(model['prior'], _EPS, 1.0 - _EPS))
    scores = np.zeros(len(n_bins), dtype=float)
    for i, nb in enumerate(n_bins):
        count = np.bincount(bins[i], minlength=int(nb))[:int(nb)].astype(float)
        weight = count / count.sum()
        rate = 1.0 / (1.0 + np.exp(-(model['logit_prior'] + model['deltas'][i])))
        rate = np.clip(rate, _EPS, 1.0 - _EPS)
        kl = rate * np.log(rate / prior) + (1.0 - rate) * np.log((1.0 - rate) / (1.0 - prior))
        scores[i] = float(np.sum(weight * kl))
    return scores


def contribution_matrix(model: dict, bins: np.ndarray, use: np.ndarray) -> np.ndarray:
    """One Naive-Bayes log-odds contribution column per selected node."""
    indexes = np.flatnonzero(use)
    return np.column_stack([model['deltas'][i][bins[i]] for i in indexes])


def _sigmoid_array(score: np.ndarray) -> np.ndarray:
    positive = score >= 0
    out = np.empty_like(score, dtype=float)
    out[positive] = 1.0 / (1.0 + np.exp(-score[positive]))
    exp_score = np.exp(score[~positive])
    out[~positive] = exp_score / (1.0 + exp_score)
    return out


def _log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), _EPS, 1.0 - _EPS)
    y = np.asarray(y, dtype=float)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def fit_weighted_logistic(
    contributions: np.ndarray,
    y: np.ndarray,
    ridge: float,
    iterations: int = 1200,
) -> dict:
    """Fit non-negative ridge weights to Naive-Bayes contribution columns."""
    X = np.asarray(contributions, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 20 or len(np.unique(y)) < 2:
        return {"intercept": float(_logit(np.mean(y))), "weights": np.zeros(X.shape[1])}

    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    active = scale > 1e-10
    Z = np.zeros_like(X)
    Z[:, active] = (X[:, active] - mean[active]) / scale[active]
    beta = np.zeros(X.shape[1], dtype=float)
    intercept = float(_logit(y.mean()))
    # Standardized columns make this conservative fixed step stable for ridge logistic
    # regression. Projection keeps a node from reversing its own Bayes evidence.
    step = 0.9 / (0.25 + ridge)
    for _ in range(iterations):
        p = _sigmoid_array(intercept + Z @ beta)
        gradient_intercept = float(np.mean(p - y))
        gradient = (Z.T @ (p - y)) / len(y) + ridge * beta
        new_intercept = intercept - step * gradient_intercept
        new_beta = np.maximum(0.0, beta - step * gradient)
        if max(abs(new_intercept - intercept), float(np.max(abs(new_beta - beta)))) < 1e-8:
            intercept, beta = new_intercept, new_beta
            break
        intercept, beta = new_intercept, new_beta

    weights = np.zeros_like(beta)
    weights[active] = beta[active] / scale[active]
    return {"intercept": float(intercept - mean @ weights), "weights": weights}


def choose_weighted_ridge(contributions: np.ndarray, y: np.ndarray) -> tuple[dict, float]:
    """Choose ridge strength on a later chronological calibration segment."""
    candidates = (0.01, 0.03, 0.10, 0.30, 1.0)
    split = max(30, int(len(y) * 0.70))
    if len(y) - split < 20 or len(np.unique(y[:split])) < 2:
        ridge = 0.10
        return fit_weighted_logistic(contributions, y, ridge), ridge
    best_ridge, best_loss = candidates[0], float("inf")
    for ridge in candidates:
        model = fit_weighted_logistic(contributions[:split], y[:split], ridge)
        p = _sigmoid_array(model["intercept"] + contributions[split:] @ model["weights"])
        loss = _log_loss(y[split:], p)
        if loss < best_loss:
            best_ridge, best_loss = ridge, loss
    return fit_weighted_logistic(contributions, y, best_ridge), best_ridge


def cross_fitted_contributions(
    y: np.ndarray,
    X: np.ndarray,
    selected: np.ndarray,
    horizon: int,
    n_bins: int,
    folds: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate chronological, out-of-fold Bayes contribution rows inside training."""
    X = np.asarray(X, dtype=float)[selected]
    n = len(y)
    cut = int(n * 0.45)
    bounds = np.linspace(cut, n, folds + 1).astype(int)
    quantiles = np.linspace(0, 1, n_bins + 1)[1:-1]
    rows, labels = [], []
    for fold in range(folds):
        t0, t1 = int(bounds[fold]), int(bounds[fold + 1])
        train_end = t0 - horizon
        if train_end < 100 or t1 - t0 < horizon * 2:
            continue
        edges = [np.unique(np.quantile(X[i, :train_end], quantiles)) for i in range(len(X))]
        counts = np.array([len(edge) + 1 for edge in edges])
        b_train = np.vstack([np.searchsorted(edges[i], X[i, :train_end]) for i in range(len(X))])
        model = fit(y[:train_end], b_train, counts)
        b_test = np.vstack([np.searchsorted(edges[i], X[i, t0:t1]) for i in range(len(X))])
        step = np.zeros(t1 - t0, dtype=bool)
        step[::horizon] = True
        rows.append(np.column_stack([model["deltas"][i][b_test[i]][step] for i in range(len(X))]))
        labels.append(y[t0:t1][step])
    if not rows:
        return np.empty((0, int(selected.sum()))), np.empty(0)
    return np.vstack(rows), np.concatenate(labels)


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


def current_weighted_forecast(
    data: pd.DataFrame,
    feats: dict,
    Δ: float,
    horizon: int,
    n_bins: int = 10,
    top_k: int | None = None,
    as_of: str | pd.Timestamp | None = None,
) -> dict:
    """Fit on completed outcomes and forecast one joint feature state.

    This is the deployment-side counterpart to :func:`walk_forward`.  The forward
    window at the forecast bar is deliberately unknown, so it is never part of the fit.
    Only labels whose entire window ended by that bar and rows with a complete feature
    panel train the node tables and their non-negative ridge weights.  ``as_of`` may
    select a historical bar; omitted, the latest complete bar is used.
    """
    if not feats:
        raise ValueError("current forecast needs at least one feature")

    names = list(feats)
    X_all = np.vstack([feats[name].to_numpy(float) for name in names])
    complete_panel = np.isfinite(X_all).all(axis=0)
    eligible_current = complete_panel.copy()
    if as_of is not None:
        eligible_current &= np.asarray(data.index <= pd.Timestamp(as_of), dtype=bool)
    current_rows = np.flatnonzero(eligible_current)
    if not len(current_rows):
        raise ValueError("no row has a complete current feature panel")
    current_index = int(current_rows[-1])

    y_all = touch_label(data, Δ, horizon)
    completed_by_forecast = np.arange(len(y_all)) + int(horizon) <= current_index
    train = completed_by_forecast & complete_panel & np.isfinite(y_all)
    if int(train.sum()) < 200:
        raise ValueError("not enough completed history for current forecast")

    y_train = y_all[train]
    X_train = X_all[:, train]
    cuts = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = [np.unique(np.quantile(X_train[index], cuts)) for index in range(len(names))]
    counts = np.array([len(edge) + 1 for edge in edges])
    bins_train = np.vstack([
        np.searchsorted(edges[index], X_train[index]) for index in range(len(names))
    ])
    model = fit(y_train, bins_train, counts)
    information = information_scores(model, bins_train, counts)
    n_top = len(names) if top_k is None else min(int(top_k), len(names))
    order = np.argsort(-information, kind="stable")
    selected = np.zeros(len(names), dtype=bool)
    selected[order[:n_top]] = True

    contributions, labels = cross_fitted_contributions(
        y_train, X_train, selected, horizon, n_bins
    )
    if len(labels) >= 40 and len(np.unique(labels)) == 2:
        weighted_model, ridge = choose_weighted_ridge(contributions, labels)
    else:
        weighted_model = {
            "intercept": model["logit_prior"],
            "weights": np.ones(int(selected.sum()), dtype=float),
        }
        ridge = None

    current_bins = np.array([
        np.searchsorted(edges[index], X_all[index, current_index])
        for index in range(len(names))
    ], dtype=int)[:, None]
    current_contributions = contribution_matrix(model, current_bins, selected)
    probability = float(_sigmoid_array(
        np.asarray([
            weighted_model["intercept"]
            + float(current_contributions[0] @ weighted_model["weights"])
        ])
    )[0])
    selected_indices = np.flatnonzero(selected)
    return {
        "probability": probability,
        "as_of": str(data.index[current_index])[:10],
        "n_observations": int(train.sum()),
        "prior": model["prior"],
        "ridge": ridge,
        "calibration_bars": int(len(labels)),
        "selected_nodes": [names[index] for index in selected_indices],
        "weights": {
            names[index]: float(weight)
            for index, weight in zip(selected_indices, weighted_model["weights"])
        },
    }


def current_weighted_surface(
    data: pd.DataFrame,
    feats: dict,
    deltas: np.ndarray,
    horizons: np.ndarray,
    n_bins: int = 10,
    top_k: int | None = None,
    as_of: str | pd.Timestamp | None = None,
    ridge_fallback: str = "model",
) -> dict:
    """Fit the complete coherent Bayes face for one latest or historical feature state."""
    if ridge_fallback not in {"model", "prior"}:
        raise ValueError("ridge_fallback must be 'model' or 'prior'")
    delta_values = np.asarray(deltas, dtype=float)
    horizon_values = np.asarray(horizons, dtype=int)
    raw = np.full((len(delta_values), len(horizon_values)), np.nan, dtype=float)
    observations = np.zeros(len(horizon_values), dtype=int)
    actual_as_of: str | None = None
    ridge_fallback_cells = 0

    for delta_index, delta in enumerate(delta_values):
        for horizon_index, horizon in enumerate(horizon_values):
            forecast = current_weighted_forecast(
                data,
                feats,
                float(delta),
                int(horizon),
                n_bins=n_bins,
                top_k=top_k,
                as_of=as_of,
            )
            raw[delta_index, horizon_index] = (
                forecast["prior"]
                if forecast["ridge"] is None and ridge_fallback == "prior"
                else forecast["probability"]
            )
            observations[horizon_index] = forecast["n_observations"]
            if forecast["ridge"] is None:
                ridge_fallback_cells += 1
            if actual_as_of is None:
                actual_as_of = forecast["as_of"]
            elif forecast["as_of"] != actual_as_of:
                raise ValueError("Bayes surface cells resolved to different as-of dates")

    if actual_as_of is None or not np.isfinite(raw).all():
        raise ValueError("Bayes probability surface is incomplete")
    return {
        "probability": coherent_probability_surface(raw, delta_values),
        "raw_probability": raw,
        "n_observations": observations,
        "as_of": actual_as_of,
        "ridge_fallback_cells": ridge_fallback_cells,
        "ridge_fallback": ridge_fallback,
    }


def _isotonic_increasing(values: np.ndarray) -> np.ndarray:
    """Least-squares non-decreasing projection via pooled adjacent violators."""
    values = np.asarray(values, dtype=float)
    means: list[float] = []
    weights: list[int] = []
    for value in values:
        means.append(float(value))
        weights.append(1)
        while len(means) >= 2 and means[-2] > means[-1]:
            weight = weights[-2] + weights[-1]
            mean = (means[-2] * weights[-2] + means[-1] * weights[-1]) / weight
            means[-2:] = [mean]
            weights[-2:] = [weight]
    return np.concatenate([
        np.full(weight, mean, dtype=float) for mean, weight in zip(means, weights)
    ])


def coherent_probability_surface(probability: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    """Project independently fitted cells onto the barrier/time nesting constraints.

    Touch probability cannot fall when the horizon grows.  At a fixed horizon it must
    rise as a downside barrier moves toward zero and fall as an upside barrier moves
    away from zero. Alternating isotonic projections impose all three constraints while
    moving the fitted probabilities as little as possible in each projection.
    """
    result = np.clip(np.asarray(probability, dtype=float), 0.0, 1.0).copy()
    deltas = np.asarray(deltas, dtype=float)
    if result.ndim != 2 or result.shape[0] != len(deltas):
        raise ValueError("probability surface must be shaped delta x horizon")
    zero = np.flatnonzero(np.isclose(deltas, 0.0))
    if len(zero) != 1 or np.any(np.diff(deltas) <= 0):
        raise ValueError("deltas must be strictly increasing and contain zero")
    zero_index = int(zero[0])

    for _ in range(200):
        previous = result.copy()
        for row in range(result.shape[0]):
            result[row] = _isotonic_increasing(result[row])
        for column in range(result.shape[1]):
            result[:zero_index + 1, column] = _isotonic_increasing(
                result[:zero_index + 1, column]
            )
            result[zero_index:, column] = -_isotonic_increasing(
                -result[zero_index:, column]
            )
        if float(np.max(np.abs(result - previous))) < 1e-10:
            break
    return np.clip(result, 0.0, 1.0)


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


# -- the walk forward ----------------------------------------------------------

def walk_forward(
    data:       pd.DataFrame,
    feats:      dict,
    families:   dict,
    Δ:      float,
    horizon:    int,
    n_bins:     int = 10,
    folds:      int = 5,
    start_frac: float = 0.5,
    top_k:      int | None = None,
    weighted:   bool = False,
) -> dict:
    """
    Expanding-window walk forward over `folds` test blocks covering the last
    (1 - start_frac) of the sample.

    Per fold, in this order:

      1. cut the training window at the fold's start, then drop its last `horizon`
         bars -- the embargo -- because their labels reach into the test block
      2. estimate bin edges, per-bin rates and the prior on what is left
      3. rank complete bin tables by training-window information gain, retain at most
         ``top_k`` distinct nodes, then keep one per family from that panel
      4. predict the test block with the edges and tables from step 2
      5. score every `horizon`-th test bar, so the scored windows do not overlap

    Returns the pooled out-of-sample predictions of both models, the prior-only
    reference, and the per-fold boundaries.
    """
    y_all = touch_label(data, Δ, horizon)
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

    output_names = ('y', 'p_all', 'p_dedup', 'p_scaled', 'p_prior', 'fold')
    if weighted:
        output_names += ('p_weighted',)
    out = {k: [] for k in output_names}
    fold_rows, selected_per_fold, kept_per_fold = [], [], []

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

        # A node is a complete conditional table, rather than its loudest decile.  This
        # rewards broad, modest gradients as their bin-level evidence accumulates, while
        # weighting a sharp regime by the share of bars on which it can help.
        information = information_scores(model, b_tr, nb)
        n_top = len(names) if top_k is None else min(int(top_k), len(names))
        order = np.argsort(-information, kind='stable')
        selected = np.zeros(len(names), dtype=bool)
        selected[order[:n_top]] = True
        selected_per_fold.append(sorted(names[i] for i in np.flatnonzero(selected)))

        # Family de-duplication is also fit strictly on this fold's training window.
        keep = np.zeros(len(names), dtype=bool)
        for family in sorted(set(fam)):
            members = [i for i in range(len(names)) if selected[i] and fam[i] == family]
            if members:
                keep[max(members, key=lambda i: information[i])] = True
        kept_per_fold.append(sorted(names[i] for i in np.flatnonzero(keep)))

        # the scale correction is fit on the training window too, so it sees no more
        # than the tables it is correcting
        a, b = platt(score(model, b_tr, keep), y_tr)

        weighted_model = None
        weighted_ridge = None
        weighted_calibration_bars = 0
        weighted_clusters: list[list[str]] = []
        if weighted:
            # Cross-fitted rows ensure the meta-model never learns weights from the
            # same observations that supplied a node's bin-rate estimate.
            d_cv, y_cv = cross_fitted_contributions(
                y_tr, X_tr, selected, horizon, n_bins
            )
            weighted_calibration_bars = int(len(y_cv))
            if len(y_cv) >= 40 and len(np.unique(y_cv)) == 2:
                weighted_model, weighted_ridge = choose_weighted_ridge(d_cv, y_cv)
            else:
                weighted_model = {
                    "intercept": model["logit_prior"],
                    "weights": np.ones(int(selected.sum()), dtype=float),
                }
                weighted_ridge = None
            selected_bins = b_tr[selected]
            similarity = redundancy.conditional_nmi_matrix(selected_bins, y_tr)
            weighted_clusters = [
                [names[np.flatnonzero(selected)[i]] for i in group]
                for group in redundancy.clusters(similarity)
            ]

        sl = slice(t0, t1)
        b_te = np.vstack([np.searchsorted(edges[i], X[i, sl]) for i in range(len(names))])

        # non-overlapping scoring bars: consecutive windows share t-1 bars of future
        step = np.zeros(t1 - t0, dtype=bool)
        step[::horizon] = True

        s_te = score(model, b_te, keep)
        out['y'].append(y_all[sl][step])
        out['p_all'].append(predict(model, b_te, selected)[step])
        out['p_dedup'].append((1.0 / (1.0 + np.exp(-s_te)))[step])
        out['p_scaled'].append((1.0 / (1.0 + np.exp(-(a * s_te + b))))[step])
        out['p_prior'].append(np.full(int(step.sum()), model['prior']))
        if weighted:
            d_test = contribution_matrix(model, b_te, selected)
            p_weighted = _sigmoid_array(
                weighted_model["intercept"] + d_test @ weighted_model["weights"]
            )
            out['p_weighted'].append(p_weighted[step])
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
            'n_selected': int(selected.sum()),
            'n_kept': int(keep.sum()),
            'platt_a': a, 'platt_b': b,
        })
        if weighted:
            fold_rows[-1].update({
                "weighted_ridge": weighted_ridge,
                "weighted_calibration_bars": weighted_calibration_bars,
                "weighted_intercept": weighted_model["intercept"],
                "weighted_nodes": {
                    names[index]: float(weight)
                    for index, weight in zip(np.flatnonzero(selected), weighted_model["weights"])
                },
                "redundancy_clusters": weighted_clusters,
            })

    if not fold_rows:
        raise ValueError('not enough history to walk forward at this horizon')

    res = {k: np.concatenate(v) for k, v in out.items()}
    res.update({'folds': fold_rows, 'selected': selected_per_fold, 'kept': kept_per_fold,
                'Δ': float(Δ), 'horizon': int(horizon),
                'nodes': names, 'n_candidates': len(names), 'top_k': top_k})
    # The prior-only forecast is constant within a fold, so it ranks nothing and its AUC
    # is undefined. Pooling the folds would give it one anyway -- a number driven purely
    # by the prior drifting between folds, which reads as discrimination and is not.
    metric_keys = [('all_nodes', 'p_all'), ('one_per_family', 'p_dedup'),
                   ('one_per_family_scaled', 'p_scaled'), ('prior_only', 'p_prior')]
    if weighted:
        metric_keys.append(('weighted_nodes', 'p_weighted'))
    res['metrics'] = {
        name: {'brier': brier(res['y'], res[key]),
               'auc': float('nan') if key == 'p_prior' else auc(res['y'], res[key]),
               'mean_predicted': float(res[key].mean())}
        for name, key in metric_keys
    }
    res['metrics']['realized_rate'] = float(res['y'].mean())
    res['metrics']['n_scored'] = int(len(res['y']))
    return res
