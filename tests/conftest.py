from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

import fixtures  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
#: Real NWB files fetched from DANDI by scripts/fetch-test-data.sh.
DANDI_CACHE = REPO / ".dandi-cache"


@pytest.fixture(scope="session")
def files(tmp_path_factory) -> dict[str, Path]:
    d = tmp_path_factory.mktemp("nwb")
    return {
        "xyzc": fixtures.neuropal_xyzc(d / "neuropal_xyzc.nwb"),
        "clef": fixtures.clef_like(d / "clef_like.nwb"),
        "subtype": fixtures.extension_subtype(d / "subtype.nwb"),
        "empty": fixtures.no_images(d / "no_images.nwb"),
        "core": fixtures.core_pynwb(d / "core.nwb"),
        "crowded": fixtures.crowded(d / "crowded.nwb"),
    }
