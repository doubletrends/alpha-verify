<!-- The PyPI project page (pyproject.toml: readme). PyPI can't resolve relative paths, so images here use full URLs;
     the GitHub README keeps relative ones. Keep "Try it" in step with the README's. -->

<div align="center">

  <img src="https://raw.githubusercontent.com/doubletrends/alpha-verify/master/docs/alphaverify-icon.svg" width="180"><br>

  <img src="https://raw.githubusercontent.com/doubletrends/alpha-verify/master/docs/badge-cuda.svg" alt="CUDA 12.0+">
  <a href="https://github.com/doubletrends/alpha-verify"><img src="https://raw.githubusercontent.com/doubletrends/alpha-verify/master/docs/badge-pypi.svg" alt="PyPI: alphaverify"></a>

  <h1>AlphaVerify - Trading-Signal Validation Framework</h1>

  <img src="https://raw.githubusercontent.com/doubletrends/alpha-verify/master/docs/stats-light.svg" width="100%">

</div>

AlphaVerify tests whether a trading signal is an edge or chance: every condition's score has to beat 95% of the same
test run on 1,000 random markets that contain no signal. The walkthrough, with the golden cross, the 200-day average
and bond yields, is on [GitHub](https://github.com/doubletrends/alpha-verify).

# Try it

**PyTorch is mandatory. A dedicated CUDA-enabled GPU is strongly recommended.**

`init` copies one of the shipped workspaces into `./workspaces`, which is where every later command looks for it.

```
python -m pip install alphaverify

alphaverify init --workspace nasdaq_daily
alphaverify measure
alphaverify compare
alphaverify validate
alphaverify select
alphaverify forecast
```

To keep workspaces anywhere else, pass `--workspaces-dir DIR` or set `$ALPHAVERIFY_WORKSPACES`.

After `validate`, the figure for every tested bin is in `workspaces/nasdaq_daily/03_validation/plot/`.

Each pipeline command also accepts `--cuda` when a CUDA-capable PyTorch installation and device are available.

# More

- [How it works](https://github.com/doubletrends/alpha-verify#how-it-works): the five stages and their mathematical form.
- [Source architecture](https://github.com/doubletrends/alpha-verify/blob/master/src/alphaverify/README.md) and
  [workspaces](https://github.com/doubletrends/alpha-verify/blob/master/workspaces/README.md).
- [`mathematics.tex`](https://github.com/doubletrends/alpha-verify/blob/master/mathematics.tex): the complete numerical specification.

AlphaVerify is research software, not investment advice.
