"""The plugin end to end, launched as vizi launches it, against the SDK's host emulator
(`examples/vizi_hostsim`, vendored in tests/_vendor; the emulator is Unix only)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fixtures import CHANNELS, C, T, X, Y, Z, expected

ROOT = Path(__file__).resolve().parents[1]
if sys.platform == "win32":
    pytest.skip("the host emulator is Unix only", allow_module_level=True)
sys.path.insert(0, str(Path(__file__).resolve().parent / "_vendor"))

from vizi_hostsim import Dataset, Host, PluginError, Segment, SeriesChoiceRequired  # noqa: E402
from vizi_plugin_sdk import pb2  # noqa: E402


@pytest.fixture(scope="module")
def plugin_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("plugin")
    subprocess.run([sys.executable, str(ROOT / "vizi_nwb_plugin.py"), f"--manifest={d}"], check=True, capture_output=True)
    return d


@pytest.fixture(scope="module")
def host(plugin_dir):
    # The host's own dataset is only for a plugin that reads host data; this one doesn't.
    unused = Dataset((1, 1, 1, 1, 1), np.dtype(np.uint8), pb2.DTYPE_U8, 1, lambda t, c: np.zeros((1, 1, 1), np.uint8))
    with Host(plugin_dir, unused) as h:
        h.check_handshake()
        yield h


@pytest.fixture(scope="module")
def grant(host):
    """vizi grants a `G` arena after every `ReaderOpened`, before the first read; each one
    supersedes the arena of the reader it replaces."""
    arena = []

    def granted(opened):
        seg = Segment.reader_arena(host.new_segment_id(), opened, host.directory)
        ack = host.supersede(arena[-1], seg) if arena else host.grant(host.add(seg).segment_id)
        assert ack.WhichOneof("outcome") == "installed", ack
        arena.append(seg.segment_id)
        return opened

    return granted


def test_a_single_image_opens_directly(host, grant, files):
    opened = grant(host.open_reader(str(files["xyzc"]), "nwb"))
    assert tuple(opened.shape_tczyx) == (1, C, Z, Y, X) and opened.dtype == pb2.DTYPE_U16
    assert list(opened.channel_names) == CHANNELS
    for c in range(C):
        np.testing.assert_array_equal(host.read_volume(opened.handle, 0, c).array, expected(0, c))
    np.testing.assert_array_equal(host.read_plane(opened.handle, 0, 2, 1).array.reshape(Y, X), expected(0, 1)[2])


def test_several_images_are_listed_and_the_choice_opens(host, grant, files):
    with pytest.raises(SeriesChoiceRequired) as listed:
        host.open_reader(str(files["clef"]), "nwb")
    ids = {e.id for e in listed.value.entries}
    assert ids == {"/acquisition/NeuroPALImageRaw", "/acquisition/CalciumImageSeries"}
    opened = grant(listed.value.choose("/acquisition/CalciumImageSeries"))
    assert tuple(opened.shape_tczyx) == (T, 1, Z, Y, X)
    np.testing.assert_array_equal(host.read_volume(opened.handle, 1, 0).array, expected(1, 0))


def test_a_file_without_images_fails_its_open(host, files):
    with pytest.raises(PluginError) as exc:
        host.open_reader(str(files["empty"]), "nwb")
    assert exc.value.kind == pb2.Error.UNSUPPORTED


def test_calibration_crosses_as_the_file_gives_it(host, grant, files):
    opened = grant(host.open_reader(str(files["xyzc"]), "nwb"))
    assert (opened.spacing_z, opened.spacing_y, opened.spacing_x) == pytest.approx((1.0, 0.33, 0.33))
    assert opened.spacing_unit == "um"
    assert opened.axis_inference == "nwb"


def test_a_series_can_be_named_with_no_listing_before_it(host, grant, files):
    # What a recent-files reopen sends.
    opened = grant(host.open_reader(str(files["clef"]), "nwb", series="/acquisition/CalciumImageSeries"))
    assert tuple(opened.shape_tczyx) == (T, 1, Z, Y, X)


def test_a_series_the_file_lacks_is_not_found(host, files):
    with pytest.raises(PluginError) as exc:
        host.open_reader(str(files["clef"]), "nwb", series="/acquisition/nothing")
    assert exc.value.kind == pb2.Error.NOT_FOUND


def test_a_file_that_is_not_hdf5_is_not_this_format(host, tmp_path):
    path = tmp_path / "fake.nwb"
    path.write_bytes(b"not an hdf5 file")
    with pytest.raises(PluginError) as exc:
        host.open_reader(str(path), "nwb")
    assert exc.value.kind == pb2.Error.INVALID_ARGUMENT


def test_a_long_walk_reports_progress_and_still_opens(host, grant, files):
    # Over the 256 objects between progress reports, so the walk really reports.
    opened = grant(host.open_reader(str(files["crowded"]), "nwb"))
    assert tuple(opened.shape_tczyx) == (1, C, Z, Y, X)
