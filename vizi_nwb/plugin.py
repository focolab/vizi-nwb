"""The vizi plugin: one reader, `nwb`, for the images in NWB files.

    python vizi_nwb_plugin.py --install     # into vizi's plugins directory
    python vizi_nwb_plugin.py --manifest=.  # just write vizi-plugin.json
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

import h5py

from vizi_plugin_sdk import (
    Cancelled,
    ReaderInfo,
    Refusal,
    SeriesChoice,
    SeriesEntry,
    cancelled,
    pb2,
    plugin,
    reader,
    report_progress,
    serve,
)

from . import __version__
from .catalog import Candidate, scan
from .volume import Volume

PLUGIN_ID = "org.focolab.nwb"
plugin(PLUGIN_ID, name="NWB", version=__version__)


@reader(
    "nwb",
    name="NWB (Neurodata Without Borders)",
    extensions=[".nwb"],
    # HDF5 files often aren't NWB: offered for them, never picked over a reader that claims them.
    fallback_extensions=[".h5", ".hdf5"],
)
class NwbReader:
    """One NWB file. Opening a file with several images answers with the list, and vizi opens
    the one the user picks on the same instance, so the file is walked once."""

    def __init__(self) -> None:
        self._path: Path | None = None
        self._file: h5py.File | None = None
        self._candidates: list[Candidate] = []
        self._volume: Volume | None = None

    def open(self, uri: str, series: str | None = None) -> ReaderInfo | SeriesChoice:
        path = _local_path(uri)
        if self._path != path:
            self.close()
            self._file = _open_hdf5(path)
            self._path = path
            try:
                self._candidates = scan(self._file, check=_progress)
            except BaseException:
                self.close()
                raise
            for cand in self._candidates:
                if cand.unusable is not None:
                    _log(f"{path.name}: not listing {cand.path}: {cand.unusable}")
        usable = [c for c in self._candidates if c.unusable is None]
        if series is None:
            if not usable:
                raise Refusal(pb2.Error.UNSUPPORTED, _nothing_to_show(path, self._candidates))
            if len(usable) == 1:
                return self._serve(usable[0])
            return SeriesChoice([_entry(c) for c in usable])
        chosen = next((c for c in self._candidates if c.path == series), None)
        if chosen is None:
            raise Refusal(pb2.Error.NOT_FOUND, f"{path.name} holds no image at {series!r}")
        if chosen.unusable is not None:
            raise Refusal(pb2.Error.UNSUPPORTED, f"{series} in {path.name} can't be shown: {chosen.unusable}")
        return self._serve(chosen)

    def read_volume(self, t: int, c: int):
        return self._live().read_volume(t, c)

    def read_plane(self, t: int, z: int, c: int):
        return self._live().read_plane(t, z, c)

    def close(self) -> None:
        file, self._file = self._file, None
        self._path, self._candidates, self._volume = None, [], None
        if file is not None:
            file.close()

    def _serve(self, cand: Candidate) -> ReaderInfo:
        self._volume = Volume(cand.dataset, cand.layout, cand.target)
        _log(f"{self._path.name}: serving {cand.path} {cand.stored_shape} as {_tczyx(cand)}: {_describe(cand)}")
        spacing = cand.spacing_zyx if cand.spacing_zyx and any(s is not None for s in cand.spacing_zyx) else None
        return ReaderInfo(
            shape_tczyx=cand.shape_tczyx,
            dtype=cand.dtype,
            spacing_zyx=spacing,
            spacing_unit="um",
            time_increment=cand.time_increment,
            time_unit="s",
            axis_inference="nwb" if cand.layout.as_specified else "nwb-heuristic",
            channel_names=cand.channel_names,
        )

    def _live(self) -> Volume:
        if self._volume is None:
            raise Refusal(pb2.Error.NOT_FOUND, "no NWB image is open")
        return self._volume


def _progress(visited: int) -> None:
    if cancelled():
        raise Cancelled("the open was canceled")
    report_progress(None, f"Looking for images: {visited} objects so far")


def _local_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme in ("http", "https", "s3", "dandi"):
        raise Refusal(pb2.Error.UNSUPPORTED, f"{uri}: only local NWB files can be opened so far")
    if parsed.scheme == "file":
        return Path(unquote(parsed.path))
    return Path(uri)


def _open_hdf5(path: Path) -> h5py.File:
    if path.is_dir():
        raise Refusal(pb2.Error.UNSUPPORTED, f"{path.name} is a directory: NWB Zarr files can't be opened yet")
    if not path.exists():
        raise FileNotFoundError(str(path))
    if not h5py.is_hdf5(path):
        raise Refusal(pb2.Error.INVALID_ARGUMENT, f"{path.name} is not an HDF5 file, so it isn't NWB")
    try:
        # Without file locking, a file on a network share or open in another program still opens.
        return h5py.File(path, "r", locking=False)
    except TypeError:  # an h5py too old to take `locking`
        return h5py.File(path, "r")


def _nothing_to_show(path: Path, candidates: list[Candidate]) -> str:
    if not candidates:
        return f"{path.name} holds no image data (no ImageSeries, Image or MultiChannelVolume)"
    reasons = "; ".join(f"{c.path}: {c.unusable}" for c in candidates)
    return f"{path.name} holds no image this reader can show ({reasons})"


def _entry(cand: Candidate) -> SeriesEntry:
    return SeriesEntry(
        cand.path,
        label=f"{cand.name} ({cand.neurodata_type})",
        shape_tczyx=cand.shape_tczyx,
        dtype=cand.dtype,
        description=_describe(cand),
    )


def _describe(cand: Candidate) -> str:
    parts = [f"{cand.layout.reason}"]
    if cand.spacing_zyx and any(s is not None for s in cand.spacing_zyx):
        z, y, x = (f"{s:g}" if s is not None else "?" for s in cand.spacing_zyx)
        parts.append(f"voxel {z} × {y} × {x} µm (z × y × x)")
    if cand.time_increment:
        parts.append(f"Δt {cand.time_increment:g} s")
    parts.extend(cand.notes)
    return ". ".join(p[0].upper() + p[1:] for p in parts) + "."


def _tczyx(cand: Candidate) -> str:
    return "(" + ", ".join(f"{a}={n}" for a, n in zip("TCZYX", cand.shape_tczyx)) + ")"


def _log(message: str) -> None:
    # vizi logs a plugin's stdout as information and its stderr as warnings, and these are neither
    # faults nor warnings.
    print(message, flush=True)


def main() -> None:
    serve()
