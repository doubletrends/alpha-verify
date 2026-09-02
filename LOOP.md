# Agent Loop — Procedure

Purely procedural. Follow these steps in order each session.
Pass `--workspace <name>` to all commands.

Every command runs against **one barrier** — the outcome declared in the workspace's
`universe.json`, or overridden with `--outcome` / `--threshold`. All output is scoped
by that barrier's event label (`dd10`, `run10`, …), so a downside sweep and an upside
sweep never overwrite each other. Check which barrier you are on before you start:
`--status` prints it in the header.

---

## 0. Orient

```
python run.py --workspace <name> --status
```

Read the table. Pick the next family with `pending > 0`. Prefer seed families before derived-only families.

Status is derived from disk, not stored: a node counts as `tested` when its xlsx
exists under the current event's directory. `skipped` nodes are listed underneath
with the reason — usually too few observations, or a barrier too rare to measure.

---

## 1. Run seeds for the chosen family

```
python run.py --workspace <name> --family <family>
```

Writes:
- `workspaces/<name>/<family>/<event>/<node_id>.xlsx` — the surface
- `workspaces/<name>/<family>/<event>/log.jsonl` — metadata + base rates

Add `--rerun` to re-run nodes that are already tested or skipped.

---

## 2. Read the results

```
python run.py --workspace <name> --read <node_id>
```

Prints significant rows from the `above` and `below` tabs. All values are deviations from the base rate — 0 = no edge.

---

## 3. Assess edge

A condition shows **candidate edge** when:
- Deviates from base rate by **> 10pp** on at least one horizon
- Deviation is **consistent across consecutive horizons** (not a single spike)
- Sample size `n` **≥ 50**

Within ±10pp across all horizons — record it and move on.

**This is a triage filter, not a result.** A peak is a maximum over ~30 bins and lands
near 10pp by construction even on a feature carrying nothing. Nothing found here counts
until it clears the shuffled null in step 7.

---

## 4. Decide: derive or exhaust

**Derive** if any of the following are true:
- A seed showed candidate edge → test parameter variants (e.g. RSI14 → RSI10, RSI18)
- Two seeds show opposing behavior → test their spread or ratio
- A seed shows edge only in extreme quantiles → test a normalized distance-from-extreme feature
- A pattern is visible but noisy → test a smoothed version

**Exhaust** if:
- All seeds sit within noise with no consistent pattern
- All reasonable derivatives have been tested and none improve on seeds
- ≥ 3 derived features tested with no candidate edge found

When exhausted, go to step 5 before picking the next family.

---

## 5. Record findings

Run `--findings` to auto-populate `best_node`, `peak_signal`, `verdict`, and `conditions` for every tested family:

```
python run.py --workspace <name> --findings
```

Then open `workspaces/<name>/findings.<event>.json` and add qualitative `notes` for this
family. The `notes` field is preserved on every subsequent `--findings` run.

Return to step 0 and pick the next family.

---

## 6. Add a derived feature

**Step 6a — Register the feature.**

If the feature is specific to this workspace, add it to `workspaces/<name>/plugin.py`:

```python
from data import features

def _my_feature(data, period): ...

features.register('my_feature', lambda d, p: _my_feature(d, p['period']))
```

If the feature is cross-asset (useful across workspaces), add it to `data/features.py`:

```python
def _my_feature(close: pd.Series, period: int) -> pd.Series: ...

register('my_feature', lambda d, p: _my_feature(d['close'], p['period']))
```

**Step 6b — Append the node** to `workspaces/<name>/universe.json` under the correct family:

```json
{
  "id": "rsi_spread_7_14",
  "family": "rsi",
  "category": "price_momentum",
  "feature": "rsi_spread",
  "params": {"fast": 7, "slow": 14},
  "data": ["ohlcv"],
  "derived_from": "rsi_7"
}
```

There is no `status` field. `universe.json` is a pure declaration and is never written
back by the pipeline; progress lives on disk.

**Step 6c — Run it:**

```
python run.py --workspace <name> --node <new_node_id>
```

Then return to step 2.

---

## 7. After all families exhausted — falsify, then combine

Once `--status` shows 0 pending across all families:

```
python run.py --workspace <name> --validate     # do this FIRST
python run.py --workspace <name> --skew
python run.py --workspace <name> --probe
python run.py --workspace <name> --backtest
python run.py --workspace <name> --findings
```

`--validate` is the step that decides whether anything found in steps 2–4 is real. It
shuffles the outcome against the feature and reads each node's peak as a quantile of
that null, Bonferroni-corrected across the whole sweep. Run it before drawing any
conclusion from a peak deviation.

`--skew` runs the mirror barrier over identical bars and reports the asymmetry. A
condition that raises the downside touch probability may simply be picking out
volatility — in which case it raises the upside barrier by just as much and carries no
directional information. Counting one barrier alone cannot tell those apart.

`--probe` evaluates current feature values and outputs a Naive Bayes combined touch
probability. `--backtest` reports calibration (does a stated 40% mean 40%?) and a Brier
skill score against the base rate. `--findings` writes the final structured record.

---

## 8. Running the mirror barrier

To sweep the same universe against the opposite barrier:

```
python run.py --workspace <name> --outcome runup --family <family>
python run.py --workspace <name> --outcome runup --validate
```

Output lands under `<family>/run10/` and `validation.run10.json`, leaving the downside
sweep untouched. Node status is tracked per event, so `--status --outcome runup` shows
that barrier's own progress.

**Scale θ to the asset and horizon.** A 10% barrier over 14 BTC days is a 21.7% event;
the same 10% over 24 NASDAQ hours is a 0.4% one, which cannot be measured at all. The
pipeline refuses to write a surface below a 5% base rate and records the reason as a
skip.

---

## Key file locations

| Path | Purpose |
|------|---------|
| `workspaces/<name>/universe.json` | Master node list + barrier declaration — add derived nodes here |
| `workspaces/<name>/findings.<event>.json` | Family verdicts — machine-generated, human-annotated |
| `workspaces/<name>/validation.<event>.json` | Per-node null-test p-values and verdicts |
| `workspaces/<name>/plugin.py` | Workspace-specific features and data sources |
| `workspaces/<name>/<family>/<event>/` | Surface xlsx files, per-family log, skip records |
| `data/features.py` | Cross-asset feature registry |
| `data/fetcher.py` | Cross-asset source registry |
| `engine/outcomes.py` | Barrier registry — the event being measured |
