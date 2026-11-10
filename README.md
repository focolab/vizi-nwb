# vizi-nwb

A [vizi](https://vizi.science) reader for the images in NWB (Neurodata Without Borders)
files: NWBelegans NeuroPAL volumes and calcium imaging series, and any other lab's NWB imaging
data. Open a `.nwb` file in vizi and the image in it opens; a file with several images asks
which one.

Needs **vizi 1.0.0 or later** (plugin protocol 1.0). Requirements are locked for Linux x86_64,
Windows x86_64 and macOS on Apple Silicon, with Python 3.11 or 3.13; it is tested with vizi 1.0.0
on Linux x86_64, macOS arm64 and Windows x86_64.

## Install

In vizi, **Tools → Plugin Manager…** installs vizi-nwb: vizi provides a Python of the version
`.python-version` names and runs `scripts/bootstrap.py` in the plugin's folder. Then
**File → Open** takes `.nwb` files.

From a clone, with no catalog entry needed:

1. Clone the repository into vizi's plugins folder as `org.focolab.nwb`. That folder is
   `.vizi/plugins` in your home folder on every system.

       git clone https://github.com/focolab/vizi-nwb ~/.vizi/plugins/org.focolab.nwb

   On Windows, in PowerShell, the target is `$HOME\.vizi\plugins\org.focolab.nwb`.
2. In that folder, build the plugin's own environment with Python 3.13, or with
   [uv](https://docs.astral.sh/uv/), which fetches that Python itself:

       python3.13 scripts/bootstrap.py                         # Linux, macOS
       py -3.13 scripts\bootstrap.py                           # Windows
       uv run --no-project --python 3.13 scripts/bootstrap.py  # any system, with uv

   Bootstrap installs exactly the pinned, hash-checked packages in `requirements/locks/` for
   your system and Python; it needs internet access to PyPI.
3. Check it. The doctor ends with `doctor: ready`.

       .venv/bin/python scripts/doctor.py                      # Linux, macOS
       .venv\Scripts\python.exe scripts\doctor.py              # Windows

4. Start vizi. In **Tools → Plugin Manager…** press **Refresh**. *NWB* is listed.

To update, run `git pull` in the folder. If bootstrap then says the `.venv` was built from a
different lock, delete `.venv` and repeat from step 2.

From a release zip: download `org.focolab.nwb-<version>.zip` from the public repository's
[Releases](https://github.com/focolab/vizi-nwb/releases). In **Tools → Plugin Manager… → Open
plugins folder**, unzip it to get a folder `org.focolab.nwb/`, then follow steps 2 to 4 in that
folder. To update, delete the folder and repeat with the new release. On macOS, Safari may unzip
the download by itself, and the extracted folder carries the quarantine flag; this route has not
been tried there.

## Use

**File → Open…** a `.nwb` file. A file with one image opens at once, and a file with several asks
which one. On macOS, under the default "All supported" filter a `.nwb` looks selectable, but Open
shows an alert that the required extension is ".tif": choose **"Plugin formats (\*.h5 \*.hdf5
\*.nwb)"** in the file-type menu first.

## What it does

- **Every image in the file is offered.** These types are found anywhere in the file:
  - the core `ImageSeries` and its subtypes (`TwoPhotonSeries`, `OnePhotonSeries`,
    `OpticalSeries`, `ImageMaskSeries`);
  - the core `Image` types (`GrayscaleImage`, `RGBImage`, `RGBAImage`);
  - ndx-multichannel-volume's `MultiChannelVolume` and `MultiChannelVolumeSeries`.

  A file with one image opens at once. A file with several lists them, keyed by their path in
  the file (`/acquisition/CalciumImageSeries`, say), with shape, type and how they'll be read.
- **Extension types are recognized from the file itself.** An NWB file embeds the schema of
  every namespace it uses. So a lab's own subtype of `ImageSeries` is found without its Python
  package, and so are the ndx-multichannel-volume types.
- **Axes are inferred, not assumed.** NWB says `(frame, x, y[, z])` for an `ImageSeries` and
  `(x, y, z, channel)` for a `MultiChannelVolume`, and real files don't always follow it.
  NWBelegans files on DANDI include `(x, y, z, c)` (000714) and `(c, z, y, x)` (001623)
  NeuroPAL images.
  - The channel axis is the one whose length matches the number of channels the imaging volume
    lists in order.
  - The spatial axes follow the series' `dimension` field when it has one, and otherwise put Z
    on the shortest axis.
  - An `ImageSeries` stored `(frame, y, x)` is recognized when its `dimension` says so.
  - Each entry in the list says how its axes were read and why. vizi logs it too, as
    `nwb-heuristic` when the result isn't the schema's order.
- **Calibration and channel names come from the file, or not at all.**
  - Voxel size comes from the imaging plane's `grid_spacing` and its unit (NWB's default unit
    is meters). Failing that it comes from a `TwoPhotonSeries`' `field_of_view` / `dimension`,
    or from an `Image`'s `resolution`.
  - The time step comes from `rate`, or from evenly spaced `timestamps`.
  - Channel names come from `order_optical_channels` when its length matches the data. RGB
    images get `red`, `green` and `blue`.
  - Anything the file doesn't record is left unknown, never filled in as 1.0.
  - NeuroPAL's `RGBW_channels` pseudocolor mapping is shown in the entry's description.
- **Samples are never altered except to fit vizi.**
  - `uint8` and `uint16` are served as they are.
  - `int8` and `int16` are widened to `float32`, exactly.
  - Larger integers and `float64` become `float32`, because a vizi reader serves at most
    `float32`. The entry says when precision can be lost.
- **Reads are lazy.** A volume or a plane is sliced straight out of the HDF5 dataset. Nothing
  else is read, and pynwb isn't used.
- **`.nwb` is this reader's. `.h5` and `.hdf5` are only fallbacks**, so a plain HDF5 file still
  goes to whichever reader claims it.

## Limits

- **Images only.** NWBelegans files also hold neuron segmentations and IDs
  (`NeuroPALSegmentation`'s `voxel_mask` and `ID_labels`).
- **Local files only.** Streaming from DANDI (remfile or fsspec) isn't supported.
- **NWB files only, not NWB Zarr** (`.nwb.zarr` directories).
- An `ImageSeries` whose frames are in external files (`format = "external"`) isn't listed.
- One image is served at a time, as whole `(Z, Y, X)` volumes, each of which has to fit in
  memory.
- `(frame, y, x)` files without a `dimension` field to say so are shown transposed. There's no
  per-file override.

## Develop

    python3.13 scripts/bootstrap.py                  # .venv, from the lock
    .venv/bin/python -m pip install 'pytest>=8' 'pynwb>=2.8'
    scripts/fetch-test-data.sh                       # optional: a real 49 MB file from DANDI 000714
    .venv/bin/python -m pytest                       # unit, emulator, real file
    .venv/bin/python vizi_nwb_plugin.py --install    # into vizi's plugins folder
    .venv/bin/python tools/write_manifest.py         # after changing a declaration
    .venv/bin/python tools/make_release.py           # dist/org.focolab.nwb-<version>.zip

The plugin is built on [vizi-plugin-sdk](https://github.com/focolab/vizi-plugin-sdk) 1.0.0,
vendored as a wheel in `vendor/`. `tools/lock.py` regenerates the locks in `requirements/locks/`
from `requirements/cpu.in`.

- `tests/fixtures.py` writes small NWB files of each layout. Every sample encodes its own
  (t, c, z, y, x), so a wrong axis order can't pass. The ndx-multichannel-volume files are
  written with plain h5py, mirroring the files on DANDI; the core ones are written with pynwb.
- `tests/test_hostsim.py` launches the plugin as vizi does, against the SDK's host emulator,
  vendored in `tests/_vendor/`. Its chooser test opens the two-image file
  `fixtures.clef_like` writes, in pytest's temporary folder. To keep one to open in vizi, with
  no test run:

      .venv/bin/python -c "import sys; sys.path.insert(0, 'tests'); import fixtures, pathlib; fixtures.clef_like(pathlib.Path('two-images.nwb'))"

- `tests/test_dandi.py` checks the real file, when it has been fetched.

| File | What it is |
|---|---|
| `vizi_nwb_plugin.py` | what vizi launches |
| `vizi_nwb/plugin.py` | the plugin: the reader declaration, open / list / open-a-series |
| `vizi_nwb/catalog.py` | the walk of the HDF5 tree: the images, their calibration, timing and channel names |
| `vizi_nwb/layout.py` | which stored axis is T, C, Z, Y and X, and why |
| `vizi_nwb/volume.py` | reading a volume or plane, and the dtype conversions |

## Credits

**This plugin is largely derivative of focolab's public NWB work.** What it knows about how
NWB imaging files, and NWBelegans files in particular, are laid out comes from:

- **[NWBelegans](https://github.com/focolab/NWBelegans)** (Daniel Sprague, focolab): the
  object names and conventions for NeuroPAL and whole-brain calcium imaging in NWB
  (`NeuroPALImageRaw`, `NeuroPALImVol`, `order_optical_channels`, `CalciumImageSeries`, ...),
  and the reading pattern of its `load_NWB.py`. See Sprague et al., *Unifying community-wide
  whole-brain imaging datasets enables robust automated neuron identification and reveals
  determinants of neuron positioning in* C. elegans, bioRxiv 2024.04.28.591397.
- **[ndx-multichannel-volume](https://github.com/focolab/ndx-multichannel-volume)**
  (Daniel Sprague, focolab; BSD-3-Clause): the `MultiChannelVolume`,
  `MultiChannelVolumeSeries`, `ImagingVolume` and `OpticalChannelReferences` types, and their
  stated axis orders.
- **[eats-worm](https://github.com/focolab/eats-worm)** (focolab; MIT): `NWB_loader.py`, whose
  lazy per-frame slicing of a `CalciumImageSeries` is the model for this reader's reads.
- **[2025-dunn-et-al-curr-biol](https://github.com/focolab/2025-dunn-et-al-curr-biol)**
  (focolab): `clefNWB.py`, which writes the channel-first NeuroPAL volumes this reader also
  recognizes.

The reader was originally written by Rex Kerr.

It's built on the [vizi plugin SDK](https://github.com/focolab/vizi-plugin-sdk). The test
data is from the [DANDI Archive](https://dandiarchive.org): dandiset 000714 (Chaudhary, Lu
lab), as converted for NWBelegans.

## License

Apache License 2.0; see `LICENSE` and `NOTICE`.
