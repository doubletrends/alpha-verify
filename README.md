<div align="center">
  <h1><strong>AlphaVerify</strong> - Multi-Asset Financial Strategy Debunker</h1>
  <img src="docs/alphaverify-icon.svg" alt="AlphaVerify icon" width="180">
  <br><br>
  <img src="https://img.shields.io/badge/Python-%E2%89%A5%203.12-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python >= 3.12">
  <img src="https://img.shields.io/badge/CUDA-%E2%89%A5%2012.0-76B900?style=for-the-badge&logo=nvidia&logoColor=white" alt="CUDA >= 12.0">
  <br><br>  
  <pre><code>pip install -e .</code></pre>
</div>

**The chart found an edge. We asked whether chance could draw it too.**

You've seen lots of them: a so-called “alpha” strategy and an equity curve that claims to beat the market. They explain the setup, promise there is no future leak, and show that it earns money.

**Most “technical” indicators are essentially astrology with better charts.**

 AlphaVerify is built to put an end to all that bullshit.

<p align="center">
<table>
  <tr>
    <td align="center" width="33%"><h2>10,000</h2>random OHLC histories</td>
    <td align="center" width="33%"><h2>19 / 20</h2>20 NASDAQ candidates</td>
    <td align="center" width="33%"><h2>1</h2>raw pass—and not proof of alpha</td>
  </tr>
</table>
<br>
</p>

The figures below are historical results from the former selection-first pipeline; they have not been regenerated with validation of all bins. Their original percentage-point score axes are historical; current shifts store `conditional_probability - baseline_probability`, and current scores are one hundredth of the former scale. A shift of 0.233 displays as 23.3%, the same 23.3-percentage-point difference.

## Example - MA Cross 50/200

The "MA Cross" heatmap looks decisive: the classic MA Cross 50/200 produces a coherent **23.3 percentage-point** upside/downside contrast. 

<p align="center">
  <a href="docs/selected_shift_surface__008__ma_cross_50_200.png"><img src="docs/selected_shift_surface__008__ma_cross_50_200.png" width="100%" alt="Observed MA Cross 50-200 condition shift surface with a 23.3 percentage-point contrast"></a>
</p>

Then the same full-grid score is applied to 10,000 fitted synthetic histories, and it turns out the **decisive edge** is just pure luck.

<p align="center">
  <a href="docs/null_histogram__008__ma_cross_50_200__bin_02.png"><img src="docs/null_histogram__008__ma_cross_50_200__bin_02.png" width="100%" alt="MA Cross 50-200 score falling well below the 95th percentile of 10,000 synthetic OHLC histories"></a>
</p>

## Try it

Run commands from the repository root. 

The distribution, CLI, and Python package are named `alphaverify`.
The repository name remains `alpha-verify`. Reinstall the editable package
after updating an existing checkout.

```
python -m pip install -e .

alphaverify measure
alphaverify compare
alphaverify validate
alphaverify select
```

Each pipeline command also accepts `--cuda` when a CUDA-capable PyTorch installation and device are available.

## How it works

| Stage | CLI Command | What it does | Mathematical form |
|---|---|---|---|
| 1 | measure | Measure conditional probabilities | `π_condition(r, δ, k, t)` |
| 2 | compare | Compare them with the baseline | `G = 100 × (π_condition − π_baseline)` |
| 3 | validate | Validate every supported bin against the null | `p̂_k < 0.05 or p̂_k ≥ 0.05` |
| 4 | select | Retain cleared bins and render their shift heatmaps | `K_selected = {k : p̂_k < 0.05}` |


Validation writes `03_validation/`; selection consumes only a current validation result and writes `04_selection/`. Every cleared bin is retained—there is no top-k ranking or secondary economic gate.

## Repository map

- [Source architecture](src/README.md) — implementation boundaries, data flow, artifact contracts, and null mechanics.
- [Test contracts](tests/README.md) — what the fast suite protects and what requires a real pipeline run.
- [Workspaces](workspaces/README.md) — experiment declarations, plugins, generated artifacts, and safe workspace changes.

AlphaVerify is research software, not investment advice.

The complete numerical specification is available as LaTeX in
[`mathematics.tex`](mathematics.tex).
