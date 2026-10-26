import os
import shutil
from pathlib import Path

import pytest

import autofit as af

directory = Path(__file__).parent


class PatchPaths(af.DirectoryPaths):
    @property
    def sym_path(self) -> Path:
        return directory / "sym_path"

    @property
    def _base_output_path(self) -> Path:
        return directory / "phase_output_path"


@pytest.fixture(name="paths")
def make_paths():
    return PatchPaths()


def test_restore(paths):
    paths.model = af.Model(af.ex.Gaussian)
    paths.save_all({}, {})

    paths.zip_remove()
    paths.restore()

    assert paths.output_path.exists()
    assert not Path(paths._zip_path).exists()

    # An unfinished search's restore claims the folder; release it as the fit's end would.
    paths._release_running_lock()
    shutil.rmtree(paths.output_path)
