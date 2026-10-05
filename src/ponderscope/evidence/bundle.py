"""Deterministic evidence bundling for official studies.

A bundle is a single compressed archive containing everything needed to audit a
run: final manifest, evidence seal, environment, tasks, raw traces/probes,
derived analysis/summaries, and the exact spec. It is deterministic (sorted
members, zeroed timestamps/ownership, gzip mtime=0) so the same run always
yields the same SHA-256. Bundles are never uploaded automatically.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from .run import (
    ANALYSIS,
    ENVIRONMENT,
    EVIDENCE,
    MANIFEST,
    PROBES,
    SUMMARY_HTML,
    SUMMARY_MD,
    TASKS,
    TRACES,
    RunStore,
    verify_run,
)

BUNDLE_FILES = (
    MANIFEST,
    EVIDENCE,
    ENVIRONMENT,
    TASKS,
    TRACES,
    PROBES,
    ANALYSIS,
    SUMMARY_MD,
    SUMMARY_HTML,
)


def bundle_run(run: str | Path, output: str | Path) -> dict[str, Any]:
    """Build a deterministic ``tar.gz`` bundle and return its SHA-256.

    Refuses to bundle a run whose evidence seal does not verify.
    """
    store = RunStore.load(run)
    report = verify_run(store.path)
    if not report["pass"]:
        raise RuntimeError(
            f"refusing to bundle {store.run_id}: evidence seal does not verify "
            f"({'; '.join(report['errors'])})"
        )
    members: dict[str, bytes] = {}
    for name in BUNDLE_FILES:
        p = store.path / name
        if p.exists():
            members[name] = p.read_bytes()
    # Include the declared spec explicitly as its own member.
    members["spec.json"] = (
        json.dumps(store.manifest.get("spec", {}), indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")

    tar_bytes = io.BytesIO()
    with tarfile.open(fileobj=tar_bytes, mode="w") as tar:
        for name in sorted(members):
            data = members[name]
            info = tarfile.TarInfo(name=f"{store.run_id}/{name}")
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            tar.addfile(info, io.BytesIO(data))

    gz = gzip.compress(tar_bytes.getvalue(), mtime=0)
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(gz)
    return {
        "run_id": store.run_id,
        "archive": str(out),
        "bytes": len(gz),
        "sha256": hashlib.sha256(gz).hexdigest(),
        "members": sorted(members),
        "format": "tar.gz",
        "deterministic": True,
    }
