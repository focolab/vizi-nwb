"""The images an NWB file holds, found by walking its HDF5 tree.

Everything here is plain h5py: every NWB group and dataset carries its `neurodata_type` as an
attribute, and the file embeds the schema of every namespace it uses under `/specifications`.
Reading types from there, rather than through pynwb, means an extension's subtype of
`ImageSeries` is recognized without the extension's Python package, and an open doesn't build
pynwb's object tree for the whole file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Callable

import h5py
import numpy as np

from . import layout as lay
from .layout import Layout, LayoutError
from .volume import UnsupportedDtype, served_dtype

#: Ancestry of the types this reader serves, for a file whose `/specifications` is missing or
#: unreadable. A file's own specifications extend it (an extension's subtypes, for instance).
BUILTIN_PARENTS = {
    "TimeSeries": "NWBDataInterface",
    "ImageSeries": "TimeSeries",
    "ImageMaskSeries": "ImageSeries",
    "OpticalSeries": "ImageSeries",
    "TwoPhotonSeries": "ImageSeries",
    "OnePhotonSeries": "ImageSeries",
    "Image": "NWBData",
    "GrayscaleImage": "Image",
    "RGBImage": "Image",
    "RGBAImage": "Image",
    "MultiChannelVolumeSeries": "TimeSeries",
    "MultiChannelVolume": "NWBDataInterface",
    "ImagingPlane": "NWBContainer",
    "ImagingVolume": "ImagingPlane",
    "OpticalChannel": "NWBContainer",
    "OpticalChannelPlus": "OpticalChannel",
    "DynamicTable": "Container",
    "ImageSegmentation": "NWBDataInterface",
    "SegmentationLabels": "NWBDataInterface",
    "PlaneSegmentation": "DynamicTable",
    "VolumeSegmentation": "PlaneSegmentation",
}

#: Groups of these types are never walked into.
_NO_IMAGES_INSIDE = ("DynamicTable", "ImageSegmentation", "SegmentationLabels", "TimeSeries")

#: Lengths in these units, as micrometers. NWB's default unit for a length is meters.
_UM_PER = {
    "m": 1e6, "meter": 1e6, "meters": 1e6, "metre": 1e6, "metres": 1e6,
    "mm": 1e3, "millimeter": 1e3, "millimeters": 1e3, "millimetre": 1e3, "millimetres": 1e3,
    "um": 1.0, "µm": 1.0, "μm": 1.0, "micrometer": 1.0, "micrometers": 1.0, "micrometre": 1.0,
    "micrometres": 1.0, "micron": 1.0, "microns": 1.0,
    "nm": 1e-3, "nanometer": 1e-3, "nanometers": 1e-3, "nanometre": 1e-3, "nanometres": 1e-3,
}
_SECONDS_PER = {"s": 1.0, "sec": 1.0, "second": 1.0, "seconds": 1.0, "ms": 1e-3, "millisecond": 1e-3, "milliseconds": 1e-3}

#: A voxel larger than this (1 mm) is taken as a unit error rather than a calibration.
_MAX_PLAUSIBLE_PITCH_UM = 1000.0


@dataclass
class Candidate:
    """One image the file holds, as a vizi series."""

    path: str
    neurodata_type: str
    kind: str
    dataset: h5py.Dataset | None
    layout: Layout | None = None
    shape_tczyx: tuple[int, int, int, int, int] | None = None
    spacing_zyx: tuple[float | None, float | None, float | None] | None = None
    time_increment: float | None = None
    channel_names: tuple[str, ...] = ()
    #: The vizi dtype the samples are served as, and the numpy dtype they're converted to.
    dtype: str | None = None
    target: np.dtype | None = None
    notes: list[str] = field(default_factory=list)
    #: Why this image can't be served, or None when it can.
    unusable: str | None = None

    @property
    def name(self) -> str:
        return self.path.rsplit("/", 1)[-1] or "/"

    @property
    def stored_shape(self) -> tuple[int, ...]:
        return tuple(self.dataset.shape) if self.dataset is not None else ()


class Types:
    """`neurodata_type` ancestry, from the file's `/specifications` and `BUILTIN_PARENTS`."""

    def __init__(self, parents: dict[str, str]) -> None:
        self.parents = parents

    @classmethod
    def of(cls, f: h5py.File) -> "Types":
        parents = dict(BUILTIN_PARENTS)
        specs = f.get("specifications")
        if isinstance(specs, h5py.Group):
            for namespace in specs.values():
                for version in namespace.values() if isinstance(namespace, h5py.Group) else ():
                    for source in version.values() if isinstance(version, h5py.Group) else ():
                        _collect_parents(_json(source), parents)
        return cls(parents)

    def is_a(self, neurodata_type: str | None, ancestor: str) -> bool:
        seen = set()
        while neurodata_type and neurodata_type not in seen:
            if neurodata_type == ancestor:
                return True
            seen.add(neurodata_type)
            neurodata_type = self.parents.get(neurodata_type)
        return False


def _json(source: h5py.Dataset) -> object:
    try:
        raw = source[()]
        return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
    except (ValueError, TypeError, OSError):
        return None


def _collect_parents(spec: object, parents: dict[str, str]) -> None:
    if isinstance(spec, dict):
        name, base = spec.get("neurodata_type_def"), spec.get("neurodata_type_inc")
        if isinstance(name, str) and isinstance(base, str):
            parents.setdefault(name, base)
        for value in spec.values():
            _collect_parents(value, parents)
    elif isinstance(spec, list):
        for value in spec:
            _collect_parents(value, parents)


def scan(f: h5py.File, *, check: Callable[[int], None] = lambda visited: None) -> list[Candidate]:
    """Every image in ``f``, in file order. ``check`` is called as the walk goes, with the number
    of objects visited so far, so a caller can report progress or stop the walk."""
    types = Types.of(f)
    found: list[Candidate] = []
    visited = 0

    def walk(group: h5py.Group) -> None:
        nonlocal visited
        for key in group:
            if group is f and key == "specifications":
                continue
            # Links are skipped: whatever they point at is found where it really is.
            if not isinstance(group.get(key, getlink=True), h5py.HardLink):
                continue
            visited += 1
            if visited % 256 == 0:
                check(visited)
            try:
                obj = group[key]
            except (KeyError, OSError):
                continue
            ntype = _text_attr(obj, "neurodata_type")
            if isinstance(obj, h5py.Dataset):
                if types.is_a(ntype, "Image"):
                    found.append(_image(obj, ntype))
                continue
            if types.is_a(ntype, "MultiChannelVolumeSeries"):
                found.append(_timeseries(obj, ntype, lay.VOLUME_SERIES, types))
            elif types.is_a(ntype, "ImageSeries"):
                found.append(_timeseries(obj, ntype, lay.IMAGE_SERIES, types))
            elif types.is_a(ntype, "MultiChannelVolume"):
                found.append(_volume(obj, ntype, types))
            elif any(types.is_a(ntype, t) for t in _NO_IMAGES_INSIDE):
                # Segmentations and traces hold no images, and there can be thousands of them
                # (NWBelegans keeps one segmentation table per time point).
                continue
            else:
                walk(obj)

    walk(f)
    return found


# --- the image types -----------------------------------------------------------------------


def _image(ds: h5py.Dataset, ntype: str) -> Candidate:
    cand = Candidate(ds.name, ntype, lay.IMAGE, ds)
    try:
        _layout(cand, lay.infer(lay.IMAGE, ds.shape))
    except LayoutError as exc:
        cand.unusable = str(exc)
        return cand
    if "C" in cand.layout.axes:
        cand.channel_names = ("red", "green", "blue", "alpha")[: cand.shape_tczyx[1]]
    resolution = _float_attr(ds, "resolution")  # pixels per centimeter
    if resolution and resolution > 0:
        pitch = 1e4 / resolution
        if pitch <= _MAX_PLAUSIBLE_PITCH_UM:
            cand.spacing_zyx = (None, pitch, pitch)
    return cand


def _timeseries(group: h5py.Group, ntype: str, kind: str, types: Types) -> Candidate:
    data = group.get("data")
    cand = Candidate(group.name, ntype, kind, data if isinstance(data, h5py.Dataset) else None)
    external = _strings(group.get("external_file"))
    if external and (cand.dataset is None or 0 in cand.dataset.shape or _text(group.get("format")) == "external"):
        listed = ", ".join(external[:3]) + (", ..." if len(external) > 3 else "")
        cand.unusable = f"its frames are in external files ({listed}), which this reader doesn't open yet"
        return cand
    if cand.dataset is None:
        cand.unusable = "it has no data"
        return cand
    plane = _linked(group, "imaging_volume" if kind == lay.VOLUME_SERIES else "imaging_plane")
    names = _channel_order(plane, types)
    dimension = _ints(group.get("dimension"))
    rgb_hint = cand.dataset.dtype == np.uint8 and cand.dataset.ndim == 4
    try:
        found = lay.infer(kind, cand.dataset.shape, n_channels=len(names) or None, dimension=dimension, rgb_hint=rgb_hint)
    except LayoutError as exc:
        cand.unusable = str(exc)
        return cand
    _layout(cand, found)
    if dimension is not None and len(dimension) >= 3 and cand.shape_tczyx[2] != dimension[2]:
        cand.notes.append(f"its dimension field says z = {dimension[2]}, the data has {cand.shape_tczyx[2]} planes")
    _channels(cand, names, plane, types)
    _spacing_from(cand, plane)
    if cand.spacing_zyx is None:
        _spacing_from_field_of_view(cand, group, dimension)
    _timing(cand, group)
    return cand


def _volume(group: h5py.Group, ntype: str, types: Types) -> Candidate:
    data = group.get("data")
    cand = Candidate(group.name, ntype, lay.VOLUME, data if isinstance(data, h5py.Dataset) else None)
    if cand.dataset is None:
        cand.unusable = "it has no data"
        return cand
    plane = _linked(group, "imaging_volume")
    names = _channel_order(plane, types)
    try:
        found = lay.infer(lay.VOLUME, cand.dataset.shape, n_channels=len(names) or None)
    except LayoutError as exc:
        cand.unusable = str(exc)
        return cand
    _layout(cand, found)
    _channels(cand, names, plane, types)
    rgbw = _ints(group.get("RGBW_channels"))
    if rgbw is not None and len(rgbw) == 4:
        mapping = ", ".join(f"{color} = {_channel_label(cand, i)}" for color, i in zip("RGBW", rgbw) if 0 <= i < cand.shape_tczyx[1])
        if mapping:
            cand.notes.append(f"NeuroPAL pseudocolor: {mapping}")
    _spacing_from(cand, plane)
    return cand


def _layout(cand: Candidate, found: Layout) -> None:
    cand.layout = found
    cand.shape_tczyx = found.shape_tczyx(cand.stored_shape)
    if 0 in cand.shape_tczyx:
        cand.unusable = "it is empty"
        return
    try:
        cand.dtype, cand.target, note = served_dtype(cand.dataset.dtype)
    except UnsupportedDtype as exc:
        cand.unusable = str(exc)
        return
    if note:
        cand.notes.append(note)


# --- calibration, timing, channels ---------------------------------------------------------


def _channel_order(plane: h5py.Group | None, types: Types) -> list[str]:
    """The channel names in data order, as an ImagingVolume's `order_optical_channels` lists
    them. A plain ImagingPlane's optical channels aren't ordered, so they don't count here."""
    if plane is None:
        return []
    for key in plane:
        obj = plane.get(key)
        if isinstance(obj, h5py.Group) and _text_attr(obj, "neurodata_type") == "OpticalChannelReferences":
            return _strings(obj.get("channels"))
    return []


def _channels(cand: Candidate, ordered: list[str], plane: h5py.Group | None, types: Types) -> None:
    c = cand.shape_tczyx[1]
    if ordered:
        if len(ordered) == c:
            cand.channel_names = tuple(ordered)
        else:
            cand.notes.append(f"the file names {len(ordered)} channels for {c} in the data, so none are named")
        return
    if plane is not None and c == 1:
        optical = [k for k in plane if types.is_a(_text_attr(plane.get(k), "neurodata_type"), "OpticalChannel")]
        if len(optical) == 1:
            cand.channel_names = (optical[0],)


def _channel_label(cand: Candidate, index: int) -> str:
    if index < len(cand.channel_names) and cand.channel_names[index]:
        return f"{cand.channel_names[index]} (channel {index + 1})"
    return f"channel {index + 1}"


def _spacing_from(cand: Candidate, plane: h5py.Group | None) -> None:
    """Voxel pitch from an imaging plane's `grid_spacing`, which is (x, y[, z])."""
    if plane is None:
        return
    spacing = plane.get("grid_spacing")
    values = _floats(spacing)
    if not values or len(values) < 2:
        return
    unit = _text_attr(spacing, "unit") or "meters"
    per = _UM_PER.get(unit.strip().lower())
    if per is None:
        cand.notes.append(f"grid_spacing is in {unit!r}, a unit this reader doesn't know, so the voxel size is unknown")
        return
    um = [v * per if v > 0 and math.isfinite(v) else None for v in values]
    if any(v is not None and v > _MAX_PLAUSIBLE_PITCH_UM for v in um):
        cand.notes.append(
            f"grid_spacing {values} {unit} would make voxels over 1 mm, so the voxel size is left unknown"
        )
        return
    x, y = um[0], um[1]
    z = um[2] if len(um) >= 3 and "Z" in cand.layout.axes else None
    cand.spacing_zyx = (z, y, x)


def _spacing_from_field_of_view(cand: Candidate, group: h5py.Group, dimension: tuple[int, ...] | None) -> None:
    """A TwoPhotonSeries without grid_spacing may give its field of view (meters) and pixel counts."""
    fov = _floats(group.get("field_of_view"))
    if not fov or not dimension or len(fov) < 2 or len(dimension) < 2:
        return
    pitch = [fov[i] * 1e6 / dimension[i] if i < len(fov) and i < len(dimension) and dimension[i] > 0 and fov[i] > 0 else None for i in range(3)]
    if any(p is not None and p > _MAX_PLAUSIBLE_PITCH_UM for p in pitch):
        return
    cand.spacing_zyx = (pitch[2] if "Z" in cand.layout.axes else None, pitch[1], pitch[0])
    cand.notes.append("voxel size from field_of_view / dimension")


def _timing(cand: Candidate, group: h5py.Group) -> None:
    start = group.get("starting_time")
    if isinstance(start, h5py.Dataset):
        rate = _float_attr(start, "rate")
        if rate and rate > 0 and math.isfinite(rate):
            cand.time_increment = 1.0 / rate
        return
    stamps = group.get("timestamps")
    if not isinstance(stamps, h5py.Dataset) or stamps.ndim != 1 or stamps.shape[0] < 2:
        return
    per = _SECONDS_PER.get((_text_attr(stamps, "unit") or "seconds").strip().lower())
    sample = np.asarray(stamps[: min(stamps.shape[0], 100_000)], dtype=np.float64)
    steps = np.diff(sample)
    if per is None or not np.all(np.isfinite(steps)) or np.any(steps <= 0):
        cand.notes.append("its timestamps aren't increasing, so the time step is unknown")
        return
    step = float(np.median(steps))
    if np.max(np.abs(steps - step)) > 0.01 * step:
        cand.notes.append(
            f"its timestamps are irregular ({steps.min() * per:.4g} s to {steps.max() * per:.4g} s apart), so no time step is given"
        )
        return
    cand.time_increment = step * per


# --- h5py helpers ----------------------------------------------------------------------------


def _linked(group: h5py.Group, key: str) -> h5py.Group | None:
    try:
        obj = group.get(key)
    except (KeyError, OSError):
        return None
    return obj if isinstance(obj, h5py.Group) else None


def _text(value: object) -> str | None:
    if isinstance(value, h5py.Dataset):
        if value.shape != ():
            return None
        value = value[()]
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _text(value.item())
    return value if isinstance(value, str) else None


def _text_attr(obj: object, name: str) -> str | None:
    try:
        return _text(obj.attrs.get(name)) if obj is not None else None
    except (OSError, AttributeError):
        return None


def _float_attr(obj: object, name: str) -> float | None:
    try:
        value = obj.attrs.get(name)
        return float(value) if value is not None else None
    except (OSError, TypeError, ValueError):
        return None


def _strings(ds: object) -> list[str]:
    if not isinstance(ds, h5py.Dataset) or ds.ndim != 1:
        return []
    return [_text(v) or "" for v in ds[()]]


def _floats(ds: object) -> list[float] | None:
    if not isinstance(ds, h5py.Dataset) or ds.ndim != 1 or not np.issubdtype(ds.dtype, np.number):
        return None
    return [float(v) for v in ds[()]]


def _ints(ds: object) -> tuple[int, ...] | None:
    if not isinstance(ds, h5py.Dataset) or ds.ndim != 1 or not np.issubdtype(ds.dtype, np.number):
        return None
    return tuple(int(v) for v in ds[()])
