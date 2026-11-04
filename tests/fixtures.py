"""Small NWB files for the tests, each sample encoding where it is.

`value(t, c, z, y, x)` is `10000 t + 1000 c + 100 z + 10 y + x`, so a volume read in the wrong
order can't match `expected(...)`. Every size differs from every other (X 7, Y 6, Z 5, C 3,
T 2) so a transposition can't hide.

The ndx-multichannel-volume files are written with plain h5py, laid out as files on DANDI are
(000714's (x, y, z, c) NeuroPAL image, 001623's (c, z, y, x) one and its (t, x, y, z) calcium
series), without the extension's package. The core ones are written with pynwb.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np

X, Y, Z, C, T = 7, 6, 5, 3, 2
CHANNELS = ["mNeptune 2.5", "CyOFP1", "mTagBFP2"]


def value(t, c, z, y, x):
    return 10000 * t + 1000 * c + 100 * z + 10 * y + x


def expected(t: int, c: int, z: int = Z, y: int = Y, x: int = X) -> np.ndarray:
    """The (Z, Y, X) volume at (t, c), as the reader should serve it."""
    zz, yy, xx = np.meshgrid(np.arange(z), np.arange(y), np.arange(x), indexing="ij")
    return value(t, c, zz, yy, xx).astype(np.uint16)


def tczyx(t=T, c=C, z=Z, y=Y, x=X) -> np.ndarray:
    tt, cc, zz, yy, xx = np.meshgrid(*(np.arange(n) for n in (t, c, z, y, x)), indexing="ij")
    return value(tt, cc, zz, yy, xx).astype(np.uint16)


def stored(axes: str, **sizes) -> np.ndarray:
    """The coordinate-encoded data stored with its axes in ``axes`` order (a subset of TCZYX)."""
    full = tczyx(**sizes)
    keep = {letter: 0 for letter in "TCZYX" if letter not in axes}
    index = tuple(keep.get(letter, slice(None)) for letter in "TCZYX")
    remaining = [letter for letter in "TCZYX" if letter not in keep]
    return np.ascontiguousarray(full[index].transpose([remaining.index(a) for a in axes]))


def _typed(obj, neurodata_type: str, namespace: str = "core") -> None:
    obj.attrs["neurodata_type"] = neurodata_type
    obj.attrs["namespace"] = namespace


def _nwb_root(f: h5py.File) -> None:
    _typed(f, "NWBFile")
    f.attrs["nwb_version"] = "2.6.0"
    for g in ("acquisition", "processing", "general/optophysiology"):
        f.require_group(g)


def _imaging_volume(f: h5py.File, name: str, channels: list[str], spacing, unit="micrometers") -> h5py.Group:
    iv = f.create_group(f"general/optophysiology/{name}")
    _typed(iv, "ImagingVolume", "ndx-multichannel-volume")
    gs = iv.create_dataset("grid_spacing", data=np.asarray(spacing, dtype=np.float64))
    if unit is not None:
        gs.attrs["unit"] = unit
    for ch in channels:
        _typed(iv.create_group(ch), "OpticalChannelPlus", "ndx-multichannel-volume")
    refs = iv.create_group("order_optical_channels")
    _typed(refs, "OpticalChannelReferences", "ndx-multichannel-volume")
    refs.create_dataset("channels", data=channels, dtype=h5py.string_dtype())
    return iv


def neuropal_xyzc(path: Path) -> Path:
    """DANDI 000714's layout: one MultiChannelVolume stored (x, y, z, channel), as the spec says."""
    with h5py.File(path, "w") as f:
        _nwb_root(f)
        _imaging_volume(f, "NeuroPALImVol", CHANNELS, [0.33, 0.33, 1.0])
        g = f.create_group("acquisition/NeuroPALImageRaw")
        _typed(g, "MultiChannelVolume", "ndx-multichannel-volume")
        g.create_dataset("data", data=stored("XYZC", t=1))
        g.create_dataset("RGBW_channels", data=np.array([0, 1, 2, -1], dtype=np.int8))
        g["imaging_volume"] = h5py.SoftLink("/general/optophysiology/NeuroPALImVol")
    return path


def clef_like(path: Path) -> Path:
    """DANDI 001623's layout: the NeuroPAL image stored (channel, z, y, x), and a
    MultiChannelVolumeSeries stored (t, x, y, z) whose `dimension` names 4 planes, not Z."""
    with h5py.File(path, "w") as f:
        _nwb_root(f)
        _imaging_volume(f, "NeuroPALImVol", CHANNELS, [0.5, 0.5, 2.0], unit="um")
        _imaging_volume(f, "CalciumImVol", ["GCaMP"], [0.5, 0.5, 2.0], unit="um")
        g = f.create_group("acquisition/NeuroPALImageRaw")
        _typed(g, "MultiChannelVolume", "ndx-multichannel-volume")
        g.create_dataset("data", data=stored("CZYX", t=1))
        g["imaging_volume"] = h5py.SoftLink("/general/optophysiology/NeuroPALImVol")
        s = f.create_group("acquisition/CalciumImageSeries")
        _typed(s, "MultiChannelVolumeSeries", "ndx-multichannel-volume")
        s.create_dataset("data", data=stored("TXYZ", c=1), chunks=(1, X, Y, 1), compression="gzip")
        start = s.create_dataset("starting_time", data=0.0)
        start.attrs["rate"] = 3.125
        s.create_dataset("dimension", data=np.array([X, Y, 4]))
        s["imaging_volume"] = h5py.SoftLink("/general/optophysiology/CalciumImVol")
    return path


def extension_subtype(path: Path) -> Path:
    """An image series of a type only the file's own /specifications says is an ImageSeries."""
    with h5py.File(path, "w") as f:
        _nwb_root(f)
        spec = {"groups": [{"neurodata_type_def": "FancyCameraSeries", "neurodata_type_inc": "ImageSeries"}]}
        f.create_dataset("specifications/ndx-fancy/0.1.0/ndx-fancy.extensions", data=json.dumps(spec))
        s = f.create_group("acquisition/Camera")
        _typed(s, "FancyCameraSeries", "ndx-fancy")
        s.create_dataset("data", data=stored("TXY", c=1, z=1))
        stamps = s.create_dataset("timestamps", data=np.array([10.0, 10.25]))
        stamps.attrs["unit"] = "seconds"
        # A segmentation table: nothing in it is an image, and the walk doesn't enter it.
        _typed(f.create_group("processing/ophys/Segmentation/Plane"), "PlaneSegmentation")
    return path


def crowded(path: Path) -> Path:
    """One NeuroPAL image among more groups than the walk takes between progress reports."""
    neuropal_xyzc(path)
    with h5py.File(path, "a") as f:
        for i in range(600):
            f.require_group(f"processing/crowd/{i}")
    return path


def no_images(path: Path) -> Path:
    with h5py.File(path, "w") as f:
        _nwb_root(f)
        ts = f.create_group("acquisition/Running")
        _typed(ts, "TimeSeries")
        ts.create_dataset("data", data=np.arange(10.0))
    return path


def core_pynwb(path: Path) -> Path:
    """Core types written by pynwb: a TwoPhotonSeries stored (t, x, y) as the schema says, one
    stored (t, y, x) that says so through its `dimension`, an ImageSeries with irregular
    timestamps, an external ImageSeries, and an Images container of a grayscale and an RGB image."""
    from pynwb import NWBHDF5IO, NWBFile
    from pynwb.base import Images
    from pynwb.image import GrayscaleImage, ImageSeries, RGBImage
    from pynwb.ophys import OpticalChannel, TwoPhotonSeries

    nwb = NWBFile(
        session_description="fixture", identifier="fixture-1", session_start_time=datetime(2026, 1, 1, tzinfo=timezone.utc)
    )
    device = nwb.create_device("microscope")
    plane = nwb.create_imaging_plane(
        name="Plane",
        optical_channel=OpticalChannel(name="GCaMP", description="green", emission_lambda=525.0),
        description="a plane",
        device=device,
        excitation_lambda=920.0,
        indicator="GCaMP6f",
        location="somewhere",
        grid_spacing=[0.8, 0.6],
        grid_spacing_unit="micrometers",
    )
    nwb.add_acquisition(
        TwoPhotonSeries(name="TwoPhotonXY", data=stored("TXY", c=1, z=1), imaging_plane=plane, rate=30.0, unit="n/a")
    )
    nwb.add_acquisition(
        TwoPhotonSeries(
            name="TwoPhotonYX",
            data=stored("TYX", c=1, z=1),
            imaging_plane=plane,
            rate=30.0,
            unit="n/a",
            dimension=[X, Y],
        )
    )
    nwb.add_acquisition(
        ImageSeries(name="Irregular", data=stored("TXY", t=3, c=1, z=1), timestamps=[0.0, 0.4, 1.5], unit="n/a")
    )
    nwb.add_acquisition(
        ImageSeries(name="Video", external_file=["video.mp4"], starting_frame=[0], format="external", rate=30.0, num_samples=100, unit="n/a")
    )
    gray = GrayscaleImage(name="gray", data=stored("XY", t=1, c=1, z=1), resolution=10000.0)
    rgb = RGBImage(name="rgb", data=stored("XYC", t=1, c=3, z=1).astype(np.uint8))
    nwb.add_acquisition(Images(name="Snapshots", images=[gray, rgb]))
    with NWBHDF5IO(str(path), "w") as io:
        io.write(nwb)
    return path
