# Domain

`domain/` defines the numbers: barrier touches, quantile bins, conditional and baseline probabilities, the bin score, the synthetic OHLC null, and the forecast combination. It performs no file I/O and never branches on whether a history is observed or synthetic. Each quantity is defined formally in [`mathematics.tex`](../../../mathematics.tex); this README covers how the code implements it.

# How-to guides

### Change what a touch, bin, or probability means

Edit `barrier.py`, then bump `MEASUREMENT_VERSION`. The version is part of the `00_cache` metadata, so observed caches rebuild themselves on the next run. It is also part of the Stage 3 `METHOD`, so earlier validation results become stale. Rerun from `measure`, because Stage 1 cubes store edges and bin assignments.

Keep `touch_tensor` (Stage 1) and `score_histories` (Stage 3) on the same `measure_histories` call. `test_scoring_parity.py` checks that the two agree.

### Change the bin score

Edit `bin_scores` in `scoring.py`, then:

1. Bump `SCORING_VERSION`.
2. Update the hand-worked expectations in `test_bin_score_reduces_all_pairs_and_horizons_with_linear_weights` and `test_full_grid_score_matches_pair_reference_with_partial_support`.
3. Keep `score_supported` separate from a zero score. A supported bin that scores zero is still tested; an unsupported bin is skipped.
4. Update the score section of `mathematics.tex` (by its owner) so that the document and the code agree.

### Change the null model

Edit `simulated_ohlc_tensor` in `validation.py` and bump `NULL_VERSION` in `pipeline/step_03_validation.py`. The ensemble must remain valid OHLCV (`low ≤ open, close ≤ high`, positive prices) and must be reproducible from its seed. `test_simulated_ensemble_is_valid_ohlcv_and_reproducible_from_its_seed` checks both. Draw it in a single `randn` call: chunked draws produce a different stream for the same seed.

Also update the statistical disclosure in [Explanation](#what-the-null-does-and-does-not-preserve).

### Add a feature

- Price- or volume-derived and reusable: follow the [tutorial](../README.md#tutorials). It must be listed in `_TORCH_OHLCV_FEATURES` so that the null recomputes it.
- Built from an auxiliary column (VIX, yields, DXY): extend `_TORCH_EXTERNAL_FEATURES` and `_external` in `features.py`. These features stay fixed in the null.
- Specific to one experiment: register it from a workspace `plugin.py` ([contract](../../../workspaces/README.md#pluginpy)). Plugin features always stay fixed in the null.

### Put a new tensor on the run's device

Use `tensor_runtime.tensor(value)`. It returns a float64 tensor on the selected device. NumPy input is always copied first, because pandas and SafeTensors can hand over read-only views. Do not call `torch.as_tensor` directly on artifact arrays.

# Explanation

### One path for observed and synthetic histories

Stage 3 asks whether a bin's observed score is large compared with scores computed the same way on histories that have no memory. That comparison holds only if both roles use literally the same code. Both therefore go through `validation.score_histories` or `score_histories_many` → `barrier.measure_histories` → `scoring.bin_scores`. The observed role may pass cached inputs (touch matrix, baseline, bin assignments, fixed edges), but these feed the same function; they do not replace it. Measurement always runs in float64 from the stored prices, and the float32 probabilities stored in Stage 1 are never scoring inputs.

### Why barriers use intraday extremes

A barrier counts as touched when a later bar's **low** or **high** reaches it, not its close. The question behind a barrier is where to place a stop or a target, and closes understate how often a level is reached intraday. On BTC daily data, closes alone put a −10%/14-day touch at 21.7% against 28.7% on the lows.

### Why bins are quantiles

Equal-width bins leave the tails of a feature almost empty, and the tails are exactly where interesting conditions sit. Quantile bins put a known, roughly equal sample behind every bin, and each bin reads as a tradable condition ("in its bottom tenth"). Features with heavy ties produce duplicate quantiles, which are removed, so such a feature can end up with fewer bins than requested. That is correct behaviour.

### Why the score compares mirror barriers

A condition that makes both +5% and −5% touches more likely is picking out volatility, not direction. The score adds up $|shift(+δ) − shift(−δ)|$, weighted linearly by distance, so it responds to an asymmetry between up and down and ignores symmetric widening. Baseline subtraction happens first, which is why market drift alone scores zero (`test_market_drift_alone_scores_zero_in_both_paths`).

### Which features the null recomputes

A synthetic path consists only of OHLCV. Features derived from it (`is_ohlcv_feature`) are recomputed on every path, so their bins move along with the synthetic prices. External, calendar, and plugin features cannot be rebuilt from a synthetic path. For those, the null keeps the observed values and edges and redraws only the prices. That still tests whether the condition predicts touches better than chance, but the condition's own history is not resampled.

### What the null does and does not preserve

The null fits one multivariate Gaussian to four log components per bar: gap, body, upper wick, and lower wick. It draws each bar independently. That preserves the mean and covariance among those four components and nothing else. There is no autocorrelation, volatility clustering, regime change, liquidity, execution, or cost. A small p-value therefore means "a memoryless simulator rarely scores this high". It does not show an out-of-sample edge. The raw 5% rule has no multiple-testing correction. Stage 4 reports Benjamini–Hochberg q-values and the count expected by chance, but it does not select by them.

### Why the forecast is naive Bayes with a joint check

`combination.naive_bayes_probability` adds each active condition's log-odds ratio against the baseline. That is exact only if the conditions are independent given the outcome, which they are not. `joint_touch_rate` measures the empirical rate on bars where every used condition held. The gap between the two estimates shows how far the independence assumption moves the forecast, though the joint sample is often thin. `nesting_violations` counts, but does not repair, cells that break the ordering implied by nested events: reaching a farther barrier implies reaching the nearer one, and a longer horizon contains the shorter one.

# Reference

### `mathematics.tex` map

| Section | Code |
|---|---|
| Central concepts; Notation | `notation.py` |
| Stage 1: measurement of probability surfaces | `barrier.py`, `features.py`, `torch_features.py` |
| Stage 2: comparison with the baseline | `shift.py`, `scoring.baseline_shifts` |
| Stage 3: Scoring a shift Surface | `scoring.bin_scores` |
| Stage 3: The synthetic-history model | `validation.py` |
| Stage 4: selection and presentation | `multiple_testing.py` |
| Stage 5: combination of active conditions | `combination.py` |
| Short reference (edges, ties, minimum dates, null fit) | `barrier.py`, `scoring.py`, `validation.py` |

### Lookups

- **Axes:** probabilities, hit counts, and shifts are `(..., barrier, bin, horizon)`. The baseline omits `bin`, bin counts omit `barrier`, and scores are `(..., bin)`.
- **Built-in features:** names are in `_TORCH_OHLCV_FEATURES` (recomputed in the null) and `_TORCH_EXTERNAL_FEATURES` in `features.py`. The parameters each one reads are in `torch_features._compute` and `features._external`.
- **Constants:** `MIN_BIN_N = 30` (`notation.py`); `REPLICATE_BATCH_SIZE = 256` and the per-bar memory costs (`validation.py`).
