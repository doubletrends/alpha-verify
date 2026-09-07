"""Convert Stage 1–4 workspace arrays from NPZ to SafeTensors.

Each source artifact is loaded through the legacy NPZ path, written through the
new SafeTensors path, reloaded for an exact payload check, then removed.  JSON
manifests and spreadsheets are intentionally untouched.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from barrierlab.infrastructure import artifact_io


WORKSPACES = ("nasdaq_daily", "btc_daily", "btc_hourly")
STAGES = ("01_surface", "02_shift", "03_selection", "04_validation")


def _matches(source: dict, converted: dict) -> bool:
    if source.keys() != converted.keys():
        return False
    for key, value in source.items():
        if key == "meta":
            if value != converted[key]:
                return False
        else:
            array = np.asarray(value)
            if array.dtype.kind in "fc":
                matches = np.array_equal(value, converted[key], equal_nan=True)
            else:
                matches = np.array_equal(value, converted[key])
            if not matches:
                return False
    return True


def main() -> None:
    converted_count = 0
    for workspace in WORKSPACES:
        for stage in STAGES:
            array_dir = Path("workspaces") / workspace / stage / "array"
            for legacy_path in sorted(array_dir.glob("*.npz")):
                target_path = legacy_path.with_suffix(".safetensors")
                if target_path.exists():
                    raise FileExistsError(f"refusing to overwrite {target_path}")
                source = artifact_io._read_npz(legacy_path)
                expected = {
                    **source,
                    "meta": json.loads(
                        json.dumps(source["meta"]).replace(".npz", ".safetensors")
                    ),
                }
                artifact_io._write_npz(
                    target_path,
                    {key: value for key, value in source.items() if key != "meta"},
                    expected["meta"],
                )
                converted = artifact_io._read_npz(target_path)
                if not _matches(expected, converted):
                    target_path.unlink()
                    raise ValueError(f"round-trip mismatch: {legacy_path}")
                legacy_path.unlink()
                converted_count += 1
    print(f"converted {converted_count} Stage 1–4 artifacts to SafeTensors")


if __name__ == "__main__":
    main()
