"""The tracked manifest is what the code declares, and says what the catalog needs."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_the_manifest_is_current():
    subprocess.run([sys.executable, str(ROOT / "tools" / "write_manifest.py"), "--check"], check=True, cwd=ROOT)


def test_the_manifest_floors_and_reader():
    data = json.loads((ROOT / "vizi-plugin.json").read_text(encoding="utf-8"))
    assert data["id"] == "org.focolab.nwb" and data["name"] == "NWB"
    assert data["engines"]["vizi"] == "1.0" and data["min_vizi"] == "1.0.0" and data["sdk_version"] == "1.0.0"
    assert data["entry"]["command"] == [".venv/bin/python", "vizi_nwb_plugin.py"]
    assert data["contributions"]["readers"] == [{
        "id": "nwb",
        "extensions": [".nwb"],
        "name": "NWB (Neurodata Without Borders)",
        "fallback_extensions": [".h5", ".hdf5"],
    }]
