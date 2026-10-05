"""Low-level atomic evidence I/O.

Evidence is append-oriented and immutable. Files are written to a temporary
sibling and atomically renamed into place. Runs are never overwritten.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, TextIO


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=path.name)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
        _fsync_dir(path.parent)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_text(path: Path, text: str) -> None:
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: Path, obj: Any) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, sort_keys=False) + "\n")


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    lines = "".join(json.dumps(row, sort_keys=False) + "\n" for row in rows)
    atomic_write_text(path, lines)


class JsonlWriter:
    """Buffered, append-only JSONL writer.

    One record is emitted per generation, not per token, so this does not sit in
    the decoding hot loop. Call :meth:`close` (or use as a context manager) to
    flush and atomically publish the file.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self._tmp: Path | None = None
        self._fh: TextIO | None = None

    def open(self) -> JsonlWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(self.path.parent), prefix=".tmp-", suffix=self.path.name)
        self._tmp = Path(tmp)
        self._fh = os.fdopen(fd, "w", encoding="utf-8")
        return self

    def append(self, row: dict[str, Any]) -> None:
        if self._fh is None:
            raise RuntimeError("JsonlWriter is not open")
        self._fh.write(json.dumps(row, sort_keys=False) + "\n")

    def close(self) -> None:
        if self._fh is None:
            return
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        assert self._tmp is not None
        os.replace(self._tmp, self.path)
        _fsync_dir(self.path.parent)
        self._fh = None
        self._tmp = None

    def __enter__(self) -> JsonlWriter:
        return self.open()

    def __exit__(self, *exc: object) -> None:
        self.close()
