"""Real NWBelegans files from DANDI. Run scripts/fetch-test-data.sh first; skipped otherwise."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from conftest import DANDI_CACHE

from vizi_plugin_sdk import ReaderInfo
from vizi_nwb.plugin import NwbReader

SUB8 = DANDI_CACHE / "000714_sub-8_ophys.nwb"


@pytest.mark.skipif(not SUB8.exists(), reason="run scripts/fetch-test-data.sh")
def test_dandi_000714_sub8_neuropal_image():
    reader = NwbReader()
    info = reader.open(str(SUB8))
    assert isinstance(info, ReaderInfo)
    assert info.shape_tczyx == (1, 4, 28, 388, 563) and info.dtype == "u16"
    assert info.spacing_zyx == (1.0, 0.33, 0.33)
    assert info.channel_names == ("mNeptune 2.5", "CyOFP1", "mTagBFP2", "Tag RFP-T")
    # The reader opens without file locking, and HDF5 refuses a second open in one process with other flags.
    with h5py.File(SUB8, "r", locking=False) as f:
        raw = f["acquisition/NeuroPALImageRaw/data"]  # (x, y, z, c)
        for c in range(4):
            np.testing.assert_array_equal(reader.read_volume(0, c), raw[:, :, :, c].transpose(2, 1, 0))
    reader.close()
