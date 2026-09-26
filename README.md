<div align="center">
  <img src="https://raw.githubusercontent.com/doubletrends/alpha-verify/master/docs/alphaverify-icon-old.svg" alt="AlphaVerify icon" width="180">
  <h1><strong>AlphaVerify</strong> - Trading-Signal Validation Framework</h1>
  <h3>Most “technical” indicators are essentially astrology with better charts.<br>  
  AlphaVerify is built to put an end to all that bullshit.</h3>
</div>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/doubletrends/alpha-verify/master/docs/stats-dark.svg">
    <img src="https://raw.githubusercontent.com/doubletrends/alpha-verify/master/docs/stats-light.svg" width="100%" alt="1,000 synthetic OHLC histories; 552 NASDAQ condition bins tested, across 57 features; 25 bins cleared, where pure chance predicts 27.6">
  </picture>
</p>

# How to look at the figures
Each figure tests one market condition. 

**Left:** while the condition holds, is the NASDAQ more likely (red) or less likely (blue) to move by a given percentage within 30 days? 
**Right:** the same test run on 1,000 random markets that contain no signal. To count as an edge, the real score (blue line) has to beat 95% of them (dashed line).

## The golden cross - folklore

Traders everywhere watch for the 50-day average to cross above the 200-day. Once it runs about 10% above, the heatmap looks decisive: the odds of a 4% drop jump, the odds of a 4% rise fall, a **24.5-point** swing. It looks like a real edge. It's folklore: random markets draw one at least this strong 8.4% of the time.

<p align="center">
  <a href="https://github.com/doubletrends/alpha-verify/blob/master/docs/bin_figure__ma_cross_50_200__bin_09.png"><img src="https://raw.githubusercontent.com/doubletrends/alpha-verify/master/docs/bin_figure__ma_cross_50_200__bin_09.png" width="100%" alt="MA Cross 50-200 bin 9: a 24.5 percentage-point contrast below the null 95th percentile"></a>
</p>

## Above the 200-day - a textbook mirror

"Stay long while price is above the 200-day" is trend following's first rule. With price 4% to 6% above it, the picture is a near-perfect mirror: rises more likely, drops much less likely, a **27.4-point** swing. Random markets draw one at least this strong 23% of the time.

<p align="center">
  <a href="https://github.com/doubletrends/alpha-verify/blob/master/docs/bin_figure__ma_ratio_200__bin_04.png"><img src="https://raw.githubusercontent.com/doubletrends/alpha-verify/master/docs/bin_figure__ma_ratio_200__bin_04.png" width="100%" alt="MA Ratio-200 bin 4: a 27.4 percentage-point contrast well inside the synthetic null"></a>
</p>

## Low bond yields - the 42-point edge

Cheap money lifts stocks; everyone knows that. When the 10-year Treasury yield sat below 1.34%, the NASDAQ was far more likely to rally 8% within a month: a **42-point** swing that grows steadily with time. It looks like a macro regime you could trade. It isn't. Random markets draw one at least this strong 7.6% of the time. Close, but not an edge.

<p align="center">
  <a href="https://github.com/doubletrends/alpha-verify/blob/master/docs/bin_figure__tnx_level__bin_01.png"><img src="https://raw.githubusercontent.com/doubletrends/alpha-verify/master/docs/bin_figure__tnx_level__bin_01.png" width="100%" alt="TNX Level bin 1: a 42 percentage-point upside contrast whose score still falls below the null 95th percentile"></a>
</p>

## Moderate yields - same signal, opposite story

Move the yield to 2.5%-2.9% and the story flips: an 8% drop becomes likelier than a rise, a **21.2-point** swing. Read alone, it's a second, bearish regime. Random markets draw one at least this strong 41% of the time.

<p align="center">
  <a href="https://github.com/doubletrends/alpha-verify/blob/master/docs/bin_figure__tnx_level__bin_06.png"><img src="https://raw.githubusercontent.com/doubletrends/alpha-verify/master/docs/bin_figure__tnx_level__bin_06.png" width="100%" alt="TNX Level bin 6: a 21.2 percentage-point downside contrast, well inside the synthetic null"></a>
</p>

# Try it

<p>
  <img src="https://img.shields.io/badge/PyTorch-%E2%89%A5%202.0-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch >= 2.0">
  <img src="https://img.shields.io/badge/CUDA-%E2%89%A5%2012.0-76B900?logo=nvidia&logoColor=white" alt="CUDA >= 12.0">
</p>

**PyTorch is mandatory. A dedicated CUDA-enabled GPU is strongly recommended.**

The distribution, CLI, and Python package are named `alphaverify`.
The repository name remains `alpha-verify`. `init` copies one of the shipped
workspaces into `./workspaces`, which is where every later command looks for it.

```
python -m pip install alphaverify

alphaverify init --workspace nasdaq_daily
alphaverify measure
alphaverify compare
alphaverify validate
alphaverify select
alphaverify forecast
```

From a checkout, `python -m pip install -e .` installs the same CLI, and `init`
is unnecessary because `workspaces/` is already there; run commands from the
repository root and reinstall the editable package after updating it. To keep
workspaces anywhere else, pass `--workspaces-dir DIR` or set `$ALPHAVERIFY_WORKSPACES`.

After `validate`, the figure for every tested bin, including the four above, is in `workspaces/nasdaq_daily/03_validation/plot/`.

Each pipeline command also accepts `--cuda` when a CUDA-capable PyTorch installation and device are available.

# How it works

| Stage | CLI Command | What it does | Mathematical form |
|---|---|---|---|
| 1 | measure | Measure conditional probabilities | `π_condition(r, δ, k, t)` |
| 2 | compare | Compare them with the baseline | `G = π_condition − π_baseline` |
| 3 | validate | Validate every supported bin against the null | `p̂_k < 0.05 or p̂_k ≥ 0.05` |
| 4 | select | Retain cleared bins, rank them by evidence, and render their figures | `K_selected = {k : p̂_k < 0.05}` |
| 5 | forecast | List which nodes cleared and with which bins; combine those active on the last stored bar, one per family, and compare with their historical joint rate | `N_cleared = {node(k) : k ∈ K_selected}`; `logit P = logit π_baseline + Σ_k (logit π_k − logit π_baseline)` |


Validation writes `03_validation/`; selection consumes only a current validation result and writes `04_selection/`; the forecast consumes only a current selection and writes `05_forecast/`. Every cleared bin is retained—there is no top-k ranking or secondary economic gate.

# Repository map

- [Source architecture](https://github.com/doubletrends/alpha-verify/blob/master/src/alphaverify/README.md) — implementation boundaries, data flow, artifact contracts, and null mechanics.
- [Workspaces](https://github.com/doubletrends/alpha-verify/blob/master/workspaces/README.md) — experiment declarations, plugins, generated artifacts, and safe workspace changes.

AlphaVerify is research software, not investment advice.

The complete numerical specification is available as LaTeX in
[`mathematics.tex`](https://github.com/doubletrends/alpha-verify/blob/master/mathematics.tex).
