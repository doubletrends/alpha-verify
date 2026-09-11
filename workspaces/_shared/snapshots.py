"""Content-addressed source-frame snapshots, explicitly called by workspaces."""

import hashlib


def snapshot(frame, cache_dir, source):
    payload = frame.to_csv(index=True).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{source}_{digest}.csv"
    path = cache_dir / filename
    if not path.exists():
        path.write_bytes(payload)
    return {"snapshot": f"00_data/{filename}", "sha256": digest}
