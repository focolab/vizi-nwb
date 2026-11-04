import pytest

from vizi_nwb import layout as lay
from vizi_nwb.layout import LayoutError, infer


def test_dandi_000714_neuropal_is_as_specified():
    found = infer(lay.VOLUME, (563, 388, 28, 4), n_channels=4)
    assert found.axes == "XYZC" and found.as_specified
    assert found.shape_tczyx((563, 388, 28, 4)) == (1, 4, 28, 388, 563)


def test_dandi_001623_neuropal_is_channel_first_numpy_order():
    found = infer(lay.VOLUME, (4, 40, 480, 2000), n_channels=4)
    assert found.axes == "CZYX" and not found.as_specified


def test_dandi_001623_calcium_series_matches_its_dimension_in_x_and_y():
    found = infer(lay.VOLUME_SERIES, (250, 2000, 480, 12), dimension=(2000, 480, 4))
    assert found.axes == "TXYZ"
    assert found.shape_tczyx((250, 2000, 480, 12)) == (250, 1, 12, 480, 2000)


def test_unknown_channel_count_takes_the_shortest_axis():
    assert infer(lay.VOLUME, (563, 388, 28, 4)).axes == "XYZC"
    assert infer(lay.VOLUME, (4, 40, 480, 2000)).axes == "CZYX"


def test_channel_count_that_matches_no_axis_falls_back_to_shortest():
    found = infer(lay.VOLUME, (100, 90, 20, 3), n_channels=5)
    assert found.axes == "XYZC" and "none has length 5" in found.reason


def test_spatial_axes_flip_so_that_z_is_shortest():
    # Channel last but the volume stored z-first: (z, y, x, c).
    assert infer(lay.VOLUME, (12, 480, 2000, 3), n_channels=3).axes == "ZYXC"


def test_five_axis_series_finds_its_channel():
    assert infer(lay.VOLUME_SERIES, (10, 64, 32, 8, 2), n_channels=2).axes == "TXYZC"
    assert infer(lay.VOLUME_SERIES, (10, 2, 8, 32, 64), n_channels=2).axes == "TCZYX"


def test_image_series_follows_the_schema_unless_dimension_says_otherwise():
    assert infer(lay.IMAGE_SERIES, (100, 512, 256)).axes == "TXY"
    assert infer(lay.IMAGE_SERIES, (100, 256, 512), dimension=(512, 256)).axes == "TYX"
    assert infer(lay.IMAGE_SERIES, (100, 512, 256, 9)).axes == "TXYZ"
    assert infer(lay.IMAGE_SERIES, (100, 512, 256, 3), rgb_hint=True).axes == "TXYC"


def test_images():
    assert infer(lay.IMAGE, (640, 480)).axes == "XY"
    assert infer(lay.IMAGE, (640, 480, 3)).axes == "XYC"
    with pytest.raises(LayoutError):
        infer(lay.IMAGE, (640, 480, 7))
    with pytest.raises(LayoutError):
        infer(lay.IMAGE_SERIES, (640, 480))


def test_dandi_000565_processed_image_puts_z_on_the_shortest_axis_and_says_it_is_uncertain():
    # (240, 5, 48, 1000): the raw NeuroPAL image beside it is (x 1000, y 240, z 48).
    found = infer(lay.VOLUME, (240, 5, 48, 1000), n_channels=4)
    assert found.axes == "XCZY" and not found.as_specified and "uncertain" in found.reason
