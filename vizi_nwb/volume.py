"""Reading one (Z, Y, X) volume or (Y, X) plane out of an NWB image dataset."""

from __future__ import annotations

import h5py
import numpy as np

from .layout import Layout

_EXACT_IN_F32 = 1 << 24


class UnsupportedDtype(ValueError):
    pass


def served_dtype(dtype: np.dtype) -> tuple[str, np.dtype, str | None]:
    """The vizi dtype ("u8", "u16" or "f32") samples of ``dtype`` are served as, the numpy dtype
    to convert to, and a note when the conversion isn't the identity."""
    dtype = np.dtype(dtype)
    if dtype == np.bool_:
        return "u8", np.dtype(np.uint8), "booleans served as 0 and 1"
    if dtype.kind == "u" and dtype.itemsize == 1:
        return "u8", np.dtype(np.uint8), None
    if dtype.kind == "u" and dtype.itemsize == 2:
        return "u16", np.dtype(np.uint16), None
    if dtype.kind in "iu" and dtype.itemsize <= 2:
        return "f32", np.dtype(np.float32), f"{dtype.name} widened to float32, exactly"
    if dtype.kind in "iu":
        return "f32", np.dtype(np.float32), (
            f"{dtype.name} served as float32: values beyond ±{_EXACT_IN_F32:,} lose precision"
        )
    if dtype.kind == "f" and dtype.itemsize <= 4:
        return "f32", np.dtype(np.float32), None if dtype.itemsize == 4 else f"{dtype.name} widened to float32"
    if dtype.kind == "f":
        return "f32", np.dtype(np.float32), f"{dtype.name} narrowed to float32 (vizi readers serve at most float32)"
    raise UnsupportedDtype(f"samples of type {dtype} can't be shown as an image")


class Volume:
    """Reads of one dataset, laid out as ``layout`` says, as samples of ``target``."""

    def __init__(self, dataset: h5py.Dataset, layout: Layout, target: np.dtype) -> None:
        self.dataset = dataset
        self.layout = layout
        self.target = np.dtype(target)

    def read_volume(self, t: int, c: int) -> np.ndarray:
        return self._read(t, c, None)

    def read_plane(self, t: int, z: int, c: int) -> np.ndarray:
        return self._read(t, c, z)

    def _read(self, t: int, c: int, z: int | None) -> np.ndarray:
        fixed = {"T": t, "C": c}
        if z is not None:
            fixed["Z"] = z
        index = []
        kept = []
        for letter in self.layout.axes:
            if letter in fixed:
                index.append(fixed[letter])
            else:
                index.append(slice(None))
                kept.append(letter)
        data = np.asarray(self.dataset[tuple(index)])
        wanted = "YX" if z is not None else "ZYX"
        for letter in wanted:
            if letter not in kept:  # an axis the data doesn't have is one sample long
                data = data[np.newaxis, ...]
                kept.insert(0, letter)
        data = data.transpose([kept.index(letter) for letter in wanted])
        return np.ascontiguousarray(data, dtype=self.target)
