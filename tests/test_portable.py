"""The plugin must start on Windows, where CI does not run: no module may
import a POSIX-only module at import time.  Each check runs in a fresh
interpreter with those modules blocked, as Windows would refuse them."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
#: Standard-library modules that do not exist on Windows.  (``posix`` is not
#: blocked: the standard library imports it itself whenever ``os.name`` is
#: ``"posix"``, so Linux cannot stand in for its absence.)
POSIX_ONLY = ("resource", "fcntl", "termios", "tty", "pty", "pwd", "grp", "crypt", "syslog")


def test_every_module_imports_without_posix_only_modules():
    blocked = "\n".join(f"sys.modules[{name!r}] = None" for name in POSIX_ONLY)
    script = f"import sys\n{blocked}\nsys.path.insert(0, {str(ROOT)!r})\n" + textwrap.dedent("""
        import importlib, pkgutil, vizi_nwb
        for mod in pkgutil.iter_modules(vizi_nwb.__path__):
            importlib.import_module(f"vizi_nwb.{mod.name}")
        import vizi_nwb_plugin  # the entry the host launches
        print("ok")
    """)
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, cwd=ROOT)
    assert done.returncode == 0 and done.stdout.strip().endswith("ok"), done.stderr
