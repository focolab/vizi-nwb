"""The release zip is the same bytes for the same files, whatever the machine."""

from __future__ import annotations

import hashlib
import importlib.util
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("make_release", ROOT / "tools" / "make_release.py")
make_release = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_release)

FILES = {"b.txt": b"b\n", "a/c.txt": b"c\n", "a.txt": b"a" * 4096}


def test_input_order_does_not_change_the_bytes(tmp_path):
    one, two = tmp_path / "one.zip", tmp_path / "two.zip"
    make_release.pack(one, "top", FILES)
    make_release.pack(two, "top", dict(reversed(list(FILES.items()))))
    assert hashlib.sha256(one.read_bytes()).digest() == hashlib.sha256(two.read_bytes()).digest()


def test_entries_are_sorted_stored_and_carry_nothing_of_the_machine(tmp_path):
    out = tmp_path / "x.zip"
    make_release.pack(out, "top", FILES)
    with zipfile.ZipFile(out) as zf:
        assert zf.namelist() == [f"top/{name}" for name in sorted(FILES)]
        for info in zf.infolist():
            assert info.date_time == make_release.EPOCH
            assert info.compress_type == zipfile.ZIP_STORED and info.create_system == 3
            assert info.external_attr == make_release.FILE_MODE << 16
