"""Path setup for running experiment scripts directly from a checkout."""

from pathlib import Path
import sys


def add_repo_paths(repo_root=None):
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[1]
    paths = [root]
    submodules_root = root / "submodules"
    if submodules_root.is_dir():
        paths.extend(path for path in submodules_root.iterdir() if path.is_dir())

    for path in reversed(paths):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)

    return root

