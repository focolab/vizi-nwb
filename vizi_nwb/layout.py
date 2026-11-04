"""Which stored axis of an NWB image dataset is time, channel, Z, Y and X.

NWB states an axis order for each image type, and files don't always follow it:

- Core `ImageSeries` (and `TwoPhotonSeries`, `OnePhotonSeries`, ...) is `(frame, x, y[, z])`,
  `Image` is `(x, y[, 3|4])`. Many files store `(frame, y, x)` instead. That can't be told from
  the data, so it is read as the schema says and the reason says so.
- ndx-multichannel-volume's `MultiChannelVolume` is `(x, y, z, channel)` and
  `MultiChannelVolumeSeries` is `(frame, x, y, z[, channel])`. NWBelegans files on DANDI also
  store `(channel, z, y, x)` (the order numpy/tifffile volumes come in), and the NWBelegans
  README documents that order for the NeuroPAL image. These are told apart from the number of
  optical channels, the series' `dimension` field and the Z axis being the shortest.

A layout is a string with one letter per stored axis, from "TCZYX".
"""

from __future__ import annotations

from dataclasses import dataclass

#: The kinds of image data a layout is inferred for.
IMAGE = "image"  # core Image: (x, y) or (x, y, 3|4)
IMAGE_SERIES = "image_series"  # core ImageSeries and its subtypes: (frame, x, y[, z])
VOLUME = "volume"  # ndx MultiChannelVolume: (x, y, z, channel)
VOLUME_SERIES = "volume_series"  # ndx MultiChannelVolumeSeries: (frame, x, y, z[, channel])


class LayoutError(ValueError):
    """The dataset's shape can't be any layout of its kind."""


@dataclass(frozen=True)
class Layout:
    axes: str
    #: Why the axes were read this way, for the series list and the plugin log.
    reason: str
    #: True when the layout is the one the schema states, unchanged by any heuristic.
    as_specified: bool

    def size(self, shape: tuple[int, ...], letter: str) -> int:
        return shape[self.axes.index(letter)] if letter in self.axes else 1

    def shape_tczyx(self, shape: tuple[int, ...]) -> tuple[int, int, int, int, int]:
        t, c, z, y, x = (self.size(shape, letter) for letter in "TCZYX")
        return (t, c, z, y, x)


def infer(
    kind: str,
    shape: tuple[int, ...],
    *,
    n_channels: int | None = None,
    dimension: tuple[int, ...] | None = None,
    rgb_hint: bool = False,
) -> Layout:
    """The layout of a dataset of ``shape``.

    ``n_channels`` is how many optical channels the file names for it, if any. ``dimension``
    is a series' recorded `(x, y[, z])` size. ``rgb_hint`` says a trailing axis of 3 or 4 holds
    color samples (8-bit data), for an `ImageSeries` whose schema has no channel axis.
    """
    shape = tuple(int(s) for s in shape)
    nd = len(shape)
    if kind == IMAGE:
        if nd == 2:
            return Layout("XY", "Image (x, y), as the NWB schema states", True)
        if nd == 3 and shape[2] in (3, 4):
            return Layout("XYC", f"Image (x, y, {shape[2]}), color samples as channels", True)
        raise LayoutError(f"an Image has 2 axes, or 3 with 3 or 4 color samples, not shape {shape}")
    if kind == IMAGE_SERIES:
        if nd not in (3, 4):
            raise LayoutError(f"an ImageSeries has 3 or 4 axes, not shape {shape}")
        # The series' `dimension` is the only evidence of a (frame, y, x) file.
        if dimension is not None and len(dimension) >= 2:
            dx, dy = int(dimension[0]), int(dimension[1])
            if (shape[2], shape[1]) == (dx, dy) and (shape[1], shape[2]) != (dx, dy):
                axes = "TYX" + ("Z" if nd == 4 else "")
                why = f"ImageSeries stored (frame, y, x): its dimension {list(dimension)} gives x = {dx}, y = {dy}"
                return Layout(axes, why, False)
        if nd == 4 and rgb_hint and shape[3] in (3, 4):
            return Layout("TXYC", f"ImageSeries (frame, x, y, {shape[3]}), 8-bit color samples as channels", False)
        if nd == 3:
            return Layout("TXY", "ImageSeries (frame, x, y), as the NWB schema states", True)
        return Layout("TXYZ", "ImageSeries (frame, x, y, z), as the NWB schema states", True)
    if kind == VOLUME:
        if nd == 3:
            spatial, why, spec = _spatial(shape, "XYZ", dimension)
            return Layout(spatial, f"MultiChannelVolume with one channel, {why}", spec)
        if nd == 4:
            return _with_channel("", shape, "MultiChannelVolume", n_channels, dimension)
        raise LayoutError(f"a MultiChannelVolume has 3 or 4 axes, not shape {shape}")
    if kind == VOLUME_SERIES:
        if nd == 4:
            spatial, why, spec = _spatial(shape[1:], "XYZ", dimension)
            return Layout("T" + spatial, f"MultiChannelVolumeSeries (frame, ...), {why}", spec)
        if nd == 5:
            return _with_channel("T", shape[1:], "MultiChannelVolumeSeries (frame, ...)", n_channels, dimension)
        raise LayoutError(f"a MultiChannelVolumeSeries has 4 or 5 axes, not shape {shape}")
    raise ValueError(f"unknown image kind {kind!r}")


def _with_channel(
    prefix: str, rest: tuple[int, ...], what: str, n_channels: int | None, dimension: tuple[int, ...] | None
) -> Layout:
    """Four axes, three spatial and one channel. The schema puts the channel last."""
    matching = [i for i, s in enumerate(rest) if n_channels and s == n_channels]
    if len(matching) == 1:
        ci = matching[0]
        why = f"the channel axis is the one of length {n_channels}, the number of optical channels"
    elif matching:
        ci = 3 if 3 in matching else matching[0]
        why = f"several axes have length {n_channels}, the number of optical channels; took axis {ci}"
    else:
        ci = min(range(4), key=lambda i: (rest[i], i != 3))
        counted = f"none has length {n_channels}, the number of optical channels; " if n_channels else ""
        why = f"{counted}took the shortest axis as the channel axis"
    spatial_shape = tuple(s for i, s in enumerate(rest) if i != ci)
    # A writer that put the channel first wrote a numpy (channel, z, y, x) volume.
    default = "ZYX" if ci == 0 else "XYZ"
    spatial, spatial_why, spatial_spec = _spatial(spatial_shape, default, dimension)
    axes = list(spatial)
    axes.insert(ci, "C")
    spec = ci == 3 and spatial_spec
    order = "as the schema states" if spec else "not the schema's (x, y, z, channel)"
    return Layout(prefix + "".join(axes), f"{what}: {why}; {spatial_why}; {order}", spec)


def _spatial(sizes: tuple[int, ...], default: str, dimension: tuple[int, ...] | None) -> tuple[str, str, bool]:
    """Three spatial axes, stored either x-first ("XYZ") or z-first ("ZYX")."""
    flipped = default[::-1]
    if dimension is not None and len(dimension) >= 2:
        x, y = int(dimension[0]), int(dimension[1])
        if (sizes[0], sizes[1]) == (x, y) and (sizes[2], sizes[1]) != (x, y):
            return "XYZ", f"x and y match the series' dimension {list(dimension)}", True
        if (sizes[2], sizes[1]) == (x, y) and (sizes[0], sizes[1]) != (x, y):
            return "ZYX", f"x and y match the series' dimension {list(dimension)}, read z-first", False
    z = default.index("Z")
    shortest = min(range(3), key=lambda i: (sizes[i], i != z))
    if sizes[shortest] == sizes[z]:
        return default, f"spatial axes read {_named(default)}", default == "XYZ"
    if shortest == 2 - z:
        return flipped, f"spatial axes read {_named(flipped)} so that z is the shortest", flipped == "XYZ"
    # Z in the middle matches no writer's convention. Keep x and y in the default's relative order.
    axes = "XZY" if default == "XYZ" else "YZX"
    return axes, f"spatial axes read {_named(axes)} so that z is the shortest, which is uncertain", False


def _named(axes: str) -> str:
    return "(" + ", ".join(axes.lower()) + ")"
