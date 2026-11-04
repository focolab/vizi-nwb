"""Build the release zip: ``dist/<plugin id>-<version>.zip``.

    .venv/bin/python tools/make_release.py

The zip holds one folder, ``org.focolab.nwb/`` (the plugin id), with what a
fresh machine needs and nothing else: the manifest, the plugin code, the Python
pin, the requirement inputs and locks, the vendored SDK wheel, bootstrap and
doctor, the README, LICENSE and NOTICE.  Only files git tracks are packed, so a
stray local file cannot ship; the manifest and every lock must be current.

The same commit gives the same bytes on any machine, so another machine can
check a release by rebuilding it and comparing the SHA-256.  Contents come from
git's objects, because a checkout may have rewritten line endings.  Entries are
stored, not deflated, because compressors differ between Python builds (some
link zlib-ng), and their order, timestamps and attributes are fixed.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHIPPED = (
    "vizi-plugin.json", "vizi_nwb_plugin.py", "vizi_nwb/", "pyproject.toml", ".python-version",
    "requirements/", "vendor/", "scripts/bootstrap.py", "scripts/doctor.py", "tools/write_manifest.py",
    "README.md", "LICENSE", "NOTICE",
)
#: The earliest time a zip can record; no file's own time reaches the archive.
EPOCH = (1980, 1, 1, 0, 0, 0)
FILE_MODE = 0o100644


def pack(out: Path, top: str, files: dict[str, bytes]) -> None:
    """Write ``files`` (path -> contents) under ``top/``, the same bytes for the same input."""
    with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as zf:
        for name in sorted(files):
            info = zipfile.ZipInfo(f"{top}/{name}", date_time=EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            info.create_system = 3  # the host OS field: otherwise Windows and Unix differ
            info.external_attr = FILE_MODE << 16
            zf.writestr(info, files[name])


def _git(*args: str) -> bytes:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, check=True).stdout


def main() -> int:
    sys.path.insert(0, str(ROOT / "tools"))
    import write_manifest

    if json.loads(write_manifest.generate()) != json.loads((ROOT / "vizi-plugin.json").read_text(encoding="utf-8")):
        sys.exit("vizi-plugin.json is stale: run tools/write_manifest.py")
    dirty = _git("status", "--porcelain", "--", *SHIPPED).decode()
    if dirty.strip():
        sys.exit(f"uncommitted changes in shipped files:\n{dirty}")
    names = _git("ls-tree", "-r", "--name-only", "-z", "HEAD", "--", *SHIPPED).decode().split("\0")
    files = {name: _git("cat-file", "blob", f"HEAD:{name}") for name in names if name}
    manifest = json.loads(files["vizi-plugin.json"])
    out = ROOT / "dist" / f"{manifest['id']}-{manifest['version']}.zip"
    out.parent.mkdir(exist_ok=True)
    pack(out, manifest["id"], files)
    print(f"wrote {out} ({len(files)} files)")
    print(f"sha256 {hashlib.sha256(out.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
