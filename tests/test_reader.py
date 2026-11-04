"""The reader class on the fixture files, without a host."""

from __future__ import annotations

import numpy as np
import pytest
from fixtures import CHANNELS, C, T, X, Y, Z, expected, stored

from vizi_plugin_sdk import ReaderInfo, Refusal, SeriesChoice, pb2
from vizi_nwb.plugin import NwbReader


def opened(path, series=None) -> tuple[NwbReader, ReaderInfo]:
    reader = NwbReader()
    info = reader.open(str(path), series)
    assert isinstance(info, ReaderInfo), info
    return reader, info


def listed(path) -> tuple[NwbReader, dict]:
    reader = NwbReader()
    choice = reader.open(str(path))
    assert isinstance(choice, SeriesChoice), choice
    return reader, {e.id: e for e in choice.entries}


def test_neuropal_xyzc_opens_directly_with_calibration_and_names(files):
    reader, info = opened(files["xyzc"])
    assert info.shape_tczyx == (1, C, Z, Y, X) and info.dtype == "u16"
    assert info.spacing_zyx == (1.0, 0.33, 0.33) and info.spacing_unit == "um"
    assert info.time_increment is None
    assert info.channel_names == tuple(CHANNELS)
    assert info.axis_inference == "nwb"
    for c in range(C):
        np.testing.assert_array_equal(reader.read_volume(0, c), expected(0, c))
        np.testing.assert_array_equal(reader.read_plane(0, 3, c), expected(0, c)[3])
    reader.close()


def test_clef_like_file_lists_both_images(files):
    reader, entries = listed(files["clef"])
    assert set(entries) == {"/acquisition/NeuroPALImageRaw", "/acquisition/CalciumImageSeries"}
    calcium = entries["/acquisition/CalciumImageSeries"]
    assert tuple(calcium.shape_tczyx) == (T, 1, Z, Y, X) and calcium.dtype == "u16"
    assert "dimension field says z = 4" in calcium.description

    info = reader.open(str(files["clef"]), "/acquisition/NeuroPALImageRaw")
    assert info.shape_tczyx == (1, C, Z, Y, X) and info.axis_inference == "nwb-heuristic"
    assert info.channel_names == tuple(CHANNELS)
    np.testing.assert_array_equal(reader.read_volume(0, 2), expected(0, 2))

    info = reader.open(str(files["clef"]), "/acquisition/CalciumImageSeries")
    assert info.time_increment == pytest.approx(0.32)
    assert info.spacing_zyx == (2.0, 0.5, 0.5)
    assert info.channel_names == ("GCaMP",)  # the one channel its imaging volume names
    for t in range(T):
        np.testing.assert_array_equal(reader.read_volume(t, 0), expected(t, 0))
    reader.close()


def test_a_named_series_opens_on_a_fresh_instance(files):
    reader, info = opened(files["clef"], "/acquisition/CalciumImageSeries")
    assert info.shape_tczyx[0] == T
    np.testing.assert_array_equal(reader.read_plane(1, 4, 0), expected(1, 0)[4])


def test_unknown_series_is_not_found(files):
    with pytest.raises(Refusal) as exc:
        NwbReader().open(str(files["clef"]), "/acquisition/Nope")
    assert exc.value.kind == pb2.Error.NOT_FOUND


def test_extension_subtype_found_through_the_files_specifications(files):
    reader, info = opened(files["subtype"])
    assert info.shape_tczyx == (T, 1, 1, Y, X)
    assert info.time_increment == pytest.approx(0.25)
    np.testing.assert_array_equal(reader.read_volume(1, 0), expected(1, 0, z=1))


def test_file_without_images_is_refused(files):
    with pytest.raises(Refusal) as exc:
        NwbReader().open(str(files["empty"]))
    assert exc.value.kind == pb2.Error.UNSUPPORTED and "no image data" in str(exc.value)


def test_not_hdf5_is_refused_as_not_this_format(tmp_path):
    bogus = tmp_path / "bogus.nwb"
    bogus.write_text("not hdf5")
    with pytest.raises(Refusal) as exc:
        NwbReader().open(str(bogus))
    assert exc.value.kind == pb2.Error.INVALID_ARGUMENT


def test_core_types_written_by_pynwb(files):
    reader, entries = listed(files["core"])
    assert set(entries) == {
        "/acquisition/TwoPhotonXY",
        "/acquisition/TwoPhotonYX",
        "/acquisition/Irregular",
        "/acquisition/Snapshots/gray",
        "/acquisition/Snapshots/rgb",
    }  # the external video isn't listed

    info = reader.open(str(files["core"]), "/acquisition/TwoPhotonXY")
    assert info.shape_tczyx == (T, 1, 1, Y, X) and info.axis_inference == "nwb"
    assert info.spacing_zyx == (None, 0.6, 0.8) and info.time_increment == pytest.approx(1 / 30)
    assert info.channel_names == ("GCaMP",)
    np.testing.assert_array_equal(reader.read_volume(1, 0), expected(1, 0, z=1))

    info = reader.open(str(files["core"]), "/acquisition/TwoPhotonYX")
    assert info.shape_tczyx == (T, 1, 1, Y, X) and info.axis_inference == "nwb-heuristic"
    np.testing.assert_array_equal(reader.read_volume(1, 0), expected(1, 0, z=1))

    info = reader.open(str(files["core"]), "/acquisition/Irregular")
    assert info.time_increment is None
    assert "irregular" in entries["/acquisition/Irregular"].description

    info = reader.open(str(files["core"]), "/acquisition/Snapshots/gray")
    assert info.shape_tczyx == (1, 1, 1, Y, X) and info.spacing_zyx == (None, 1.0, 1.0)

    info = reader.open(str(files["core"]), "/acquisition/Snapshots/rgb")
    assert info.shape_tczyx == (1, 3, 1, Y, X) and info.dtype == "u8"
    assert info.channel_names == ("red", "green", "blue")
    rgb = stored("XYC", t=1, c=3, z=1).astype(np.uint8)
    np.testing.assert_array_equal(reader.read_volume(0, 2)[0], rgb[:, :, 2].T)
    reader.close()


def test_what_the_reader_serves_is_information_not_a_warning(files, capsys):
    reader = NwbReader()
    reader.open(str(files["xyzc"]))
    reader.close()
    written = capsys.readouterr()
    assert "serving /acquisition/NeuroPALImageRaw" in written.out and written.err == ""
