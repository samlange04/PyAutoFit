import pickle
from pathlib import Path

import pytest

import autofit as af
from autonerves.conf import output_path_for_test
from autofit.non_linear.paths.null import NullPaths


def test_null_paths():
    search = af.DynestyStatic()

    assert isinstance(search.paths, NullPaths)


class TestPathDecorator:
    @staticmethod
    def assert_paths_as_expected(paths):
        assert paths.name == "name"
        assert paths.path_prefix == ""

    def test_with_arguments(self):
        search = af.m.MockSearch()
        search.paths = af.DirectoryPaths(name="name")

        self.assert_paths_as_expected(search.paths)

    def test_positional(self):
        search = af.m.MockSearch("name")
        paths = search.paths

        assert paths.name == "name"

    def test_paths_argument(self):
        search = af.m.MockSearch()
        search.paths = af.DirectoryPaths(name="name")
        self.assert_paths_as_expected(search.paths)

    def test_combination_argument(self):
        search = af.m.MockSearch(
            "other",
        )
        search.paths = af.DirectoryPaths(name="name")
        self.assert_paths_as_expected(search.paths)


output_path = Path(__file__).parent / "path"


@pytest.fixture(name="model")
def make_model():
    return af.Model(af.ex.Gaussian)


@output_path_for_test(output_path)
def test_identifier_file(model):
    paths = af.DirectoryPaths()
    paths.model = model
    paths.search = af.DynestyStatic()
    paths.save_all({}, {})

    assert (output_path / paths.identifier / ".identifier").exists()


def test_serialize(model):
    paths = af.DirectoryPaths()
    paths.model = model

    pickled_paths = pickle.loads(pickle.dumps(paths))

    assert pickled_paths.model is not None


def test_unique_tag():
    paths = af.DirectoryPaths(unique_tag="unique_tag")
    assert "unique_tag" in paths.output_path.parts


class TestTestModeOutputPath:
    """
    PYAUTO_TEST_MODE must namespace the AutoFit output path so smoke
    runs cannot poison the cache for a later real run at the same
    paths. Regression for the "Fit Already Completed" silent skip.
    """

    def test_output_path_contains_test_mode_segment_when_env_set(
        self, monkeypatch
    ):
        monkeypatch.setenv("PYAUTO_TEST_MODE", "2")
        paths = af.DirectoryPaths(name="name", path_prefix="prefix")
        assert "test_mode" in paths.output_path.parts

    def test_output_path_excludes_segment_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
        paths = af.DirectoryPaths(name="name", path_prefix="prefix")
        assert "test_mode" not in paths.output_path.parts

    def test_output_path_excludes_segment_when_level_zero(self, monkeypatch):
        monkeypatch.setenv("PYAUTO_TEST_MODE", "0")
        paths = af.DirectoryPaths(name="name", path_prefix="prefix")
        assert "test_mode" not in paths.output_path.parts

    def test_make_path_contains_segment_when_env_set(self, monkeypatch):
        monkeypatch.setenv("PYAUTO_TEST_MODE", "2")
        paths = af.DirectoryPaths(name="name", path_prefix="prefix")
        assert "test_mode" in paths._make_path().parts

    def test_make_path_excludes_segment_when_env_unset(self, monkeypatch):
        monkeypatch.delenv("PYAUTO_TEST_MODE", raising=False)
        paths = af.DirectoryPaths(name="name", path_prefix="prefix")
        assert "test_mode" not in paths._make_path().parts

    def test_make_path_excludes_segment_when_level_zero(self, monkeypatch):
        monkeypatch.setenv("PYAUTO_TEST_MODE", "0")
        paths = af.DirectoryPaths(name="name", path_prefix="prefix")
        assert "test_mode" not in paths._make_path().parts


def test__preserve_in_zip__file_survives_restore(tmp_path):
    import zipfile

    paths = af.DirectoryPaths(name="preserve_test", path_prefix=str(tmp_path))
    # The output directory is the primary store, so the loose copy stays put;
    # ``test__preserve_in_zip__removes_the_loose_copy_under_remove_files`` covers
    # the other setting.
    paths.remove_files = False

    files_path = Path(paths._files_path)
    files_path.mkdir(parents=True, exist_ok=True)
    (files_path / "samples_summary.json").write_text("{}")

    paths.zip_remove()
    assert Path(paths._zip_path).exists()

    # A post-completion cache write: on disk but not in the zip.
    cache_file = files_path / "cache_artifact.json"
    files_path.mkdir(parents=True, exist_ok=True)
    cache_file.write_text('{"cached": true}')

    paths.preserve_in_zip(cache_file)

    with zipfile.ZipFile(paths._zip_path) as f:
        assert "files/cache_artifact.json" in f.namelist()

    # Idempotent — appending again must not duplicate the member.
    paths.preserve_in_zip(cache_file)
    with zipfile.ZipFile(paths._zip_path) as f:
        assert f.namelist().count("files/cache_artifact.json") == 1

    # The restore cycle (rmtree + re-extract) keeps the preserved file.
    paths.restore()
    assert cache_file.exists()


def test__preserve_in_zip__removes_the_loose_copy_under_remove_files(tmp_path):
    """
    Under ``remove_files`` the zip is the search's only store, so once the file
    is a member the loose copy — and every directory it emptied, up to and
    including ``output_path`` — is removed. Left behind, that directory shadows
    the zip for ``Aggregator.from_directory`` and, carrying no ``.completed``
    file, silently drops the search under ``completed_only``.
    """
    import zipfile

    paths = af.DirectoryPaths(name="preserve_remove_test", path_prefix=str(tmp_path))
    paths.remove_files = True

    files_path = Path(paths._files_path)
    files_path.mkdir(parents=True, exist_ok=True)
    (files_path / "samples_summary.json").write_text("{}")

    paths.zip_remove()
    assert Path(paths._zip_path).exists()
    assert not Path(paths.output_path).exists()

    # A post-completion cache write recreates ``<search>/files/``.
    cache_file = Path(paths._files_path) / "cache_artifact.json"
    cache_file.write_text('{"cached": true}')

    paths.preserve_in_zip(cache_file)

    with zipfile.ZipFile(paths._zip_path) as f:
        assert f.read("files/cache_artifact.json") == b'{"cached": true}'

    assert not cache_file.exists()
    assert not files_path.exists()
    assert not Path(paths.output_path).exists()

    # The restore cycle still brings it back from the zip.
    paths.restore()
    assert cache_file.exists()


def test__preserve_in_zip__keeps_the_loose_copy_without_remove_files(tmp_path):
    """
    With ``remove_files`` unset the output directory is the primary store, so
    the loose copy is left where the writer put it.
    """
    paths = af.DirectoryPaths(name="preserve_keep_test", path_prefix=str(tmp_path))
    paths.remove_files = False

    files_path = Path(paths._files_path)
    files_path.mkdir(parents=True, exist_ok=True)
    (files_path / "samples_summary.json").write_text("{}")

    paths.zip_remove()
    assert Path(paths._zip_path).exists()

    cache_file = Path(paths._files_path) / "cache_artifact.json"
    cache_file.write_text('{"cached": true}')

    paths.preserve_in_zip(cache_file)

    assert cache_file.exists()
    assert cache_file.read_text() == '{"cached": true}'


def test__preserve_in_zip__replaces_stale_member(tmp_path):
    import zipfile

    paths = af.DirectoryPaths(name="preserve_replace_test", path_prefix=str(tmp_path))

    files_path = Path(paths._files_path)
    files_path.mkdir(parents=True, exist_ok=True)
    (files_path / "samples_summary.json").write_text("{}")

    cache_file = files_path / "cache_artifact.json"
    cache_file.write_text('{"cached": "stale"}')

    paths.zip_remove()
    assert Path(paths._zip_path).exists()

    # The cache was invalidated and recomputed: the member is already in the
    # zip, but its content has changed.
    files_path.mkdir(parents=True, exist_ok=True)
    cache_file.write_text('{"cached": "recomputed"}')
    paths.preserve_in_zip(cache_file)

    with zipfile.ZipFile(paths._zip_path) as f:
        assert f.namelist().count("files/cache_artifact.json") == 1
        assert f.read("files/cache_artifact.json") == b'{"cached": "recomputed"}'
        # The rewrite must not lose the members it copied across.
        assert f.read("files/samples_summary.json") == b"{}"

    # The restore cycle yields the recomputed bytes, not the stale ones.
    paths.restore()
    assert cache_file.read_text() == '{"cached": "recomputed"}'


def test__preserve_in_zip__identical_content_does_not_rewrite(tmp_path):
    paths = af.DirectoryPaths(name="preserve_identical_test", path_prefix=str(tmp_path))

    files_path = Path(paths._files_path)
    files_path.mkdir(parents=True, exist_ok=True)
    cache_file = files_path / "cache_artifact.json"
    cache_file.write_text('{"cached": true}')

    paths.zip_remove()
    before = Path(paths._zip_path).read_bytes()

    # Re-preserving a file whose content is unchanged leaves the archive
    # untouched, so the common resume path never pays for a rewrite.
    files_path.mkdir(parents=True, exist_ok=True)
    cache_file.write_text('{"cached": true}')

    paths.preserve_in_zip(cache_file)

    assert Path(paths._zip_path).read_bytes() == before


def test__preserve_in_zip__no_zip_is_a_no_op(tmp_path):
    paths = af.DirectoryPaths(name="preserve_noop_test", path_prefix=str(tmp_path))

    files_path = Path(paths._files_path)
    files_path.mkdir(parents=True, exist_ok=True)
    cache_file = files_path / "cache_artifact.json"
    cache_file.write_text('{"cached": true}')

    paths.preserve_in_zip(cache_file)

    assert not Path(paths._zip_path).exists()
    assert cache_file.exists()


def _completed_archive(tmp_path, name="readonly_test"):
    """A finished search's archive: files + ``.completed`` zipped, folder removed."""
    paths = af.DirectoryPaths(name=name, path_prefix=str(tmp_path))
    paths.remove_files = True
    files_path = Path(paths._files_path)
    files_path.mkdir(parents=True, exist_ok=True)
    (files_path / "samples_summary.json").write_text("{}")
    paths.completed()
    paths.zip_remove()
    assert Path(paths._zip_path).exists()
    assert not Path(paths.output_path).exists()
    return paths


def test__restore__completed_archive_is_replayed_read_only(tmp_path):
    import zipfile

    from autofit.non_linear.paths import abstract

    paths = _completed_archive(tmp_path)
    base = Path(paths.output_path)
    zip_path = Path(paths._zip_path)
    before = (zip_path.stat().st_mtime_ns, zip_path.stat().st_size)

    paths.restore()

    # Extracted into a private folder beside the zip, which stays; the base folder is untouched.
    replay = Path(paths.output_path)
    assert replay != base and ".replay-" in replay.name
    assert paths.is_complete
    assert (replay / ".completed").exists()
    assert not base.exists()
    assert zip_path.exists()
    assert paths._archive_is_complete
    assert Path(paths._zip_path) == zip_path

    paths.zip_remove()

    # Kept for the process (later stages read and write caches here), dropped at exit.
    assert replay.exists()
    assert not paths._archive_is_complete
    assert (zip_path.stat().st_mtime_ns, zip_path.stat().st_size) == before
    with zipfile.ZipFile(zip_path) as f:
        assert ".completed" in f.namelist()
    abstract._cleanup_replay_dirs()
    assert not replay.exists()


def test__restore__dead_process_replay_folder_is_removed(tmp_path):
    import socket

    paths = _completed_archive(tmp_path)
    base = Path(paths.output_path)
    dead = base.parent / f"{base.name}.replay-{socket.gethostname()}-999999999"
    dead.mkdir()
    alive_other_host = base.parent / f"{base.name}.replay-elsewhere-1"
    alive_other_host.mkdir()

    paths.restore()

    assert not dead.exists()
    assert alive_other_host.exists()  # cannot be checked from here, left alone
    paths.zip_remove()


def test__restore__unfinished_archive_round_trips_as_before(tmp_path):
    paths = af.DirectoryPaths(name="resume_test", path_prefix=str(tmp_path))
    paths.remove_files = True
    files_path = Path(paths._files_path)
    files_path.mkdir(parents=True, exist_ok=True)
    (files_path / "samples_summary.json").write_text("{}")
    paths.zip_remove()

    paths.restore()

    assert not Path(paths._zip_path).exists()
    assert not paths._archive_is_complete

    paths.zip_remove()
    assert Path(paths._zip_path).exists()


def test__restore__two_processes_replaying_one_archive_do_not_share_a_folder(tmp_path, monkeypatch):
    from autofit.non_linear.paths import abstract

    first = _completed_archive(tmp_path)
    first.restore()
    first_dir = Path(first.output_path)
    assert (first_dir / ".completed").exists()

    # Another process (a different pid) replaying the same archive.
    monkeypatch.setattr(abstract.os, "getpid", lambda: 999999998)
    second = af.DirectoryPaths(name="readonly_test", path_prefix=str(tmp_path))
    second.remove_files = True
    second.restore()
    second_dir = Path(second.output_path)
    assert second_dir != first_dir and (second_dir / ".completed").exists()
    second.zip_remove()
    monkeypatch.undo()

    # The second's replay and cleanup left the first's copy alone (the race of 2026-10-26),
    # and every Paths of this process for the search resolves to the first's folder.
    assert (first_dir / "files" / "samples_summary.json").exists()
    third = af.DirectoryPaths(name="readonly_test", path_prefix=str(tmp_path))
    assert Path(third.output_path) == first_dir
    first.zip_remove()
    abstract._cleanup_replay_dirs()
    assert not first_dir.exists() and not second_dir.exists()


def test__restore__unfinished_search_is_locked_against_a_second_process(tmp_path):
    import json
    import os
    from autofit.non_linear.paths.abstract import SearchInProgressError

    paths = af.DirectoryPaths(name="running_test", path_prefix=str(tmp_path))
    paths.remove_files = True

    # A fresh start claims the folder; the lock names this process.
    paths.restore()
    lock = paths._running_lock_path
    assert lock.is_dir()
    holder = json.loads((lock / "holder").read_text())
    assert holder["pid"] == os.getpid()

    # A second process on the same folder is refused while the holder is alive. The holder
    # file is rewritten with the parent's pid (and start time, which guards against a reused
    # pid): a lock naming this very process would be taken over (an earlier fit here that
    # raised before releasing it).
    from autofit.non_linear.paths.abstract import _pid_start_time

    (lock / "holder").write_text(
        json.dumps({**holder, "pid": os.getppid(), "pid_start": _pid_start_time(os.getppid())})
    )
    other = af.DirectoryPaths(name="running_test", path_prefix=str(tmp_path))
    with pytest.raises(SearchInProgressError):
        other.restore()
    (lock / "holder").write_text(json.dumps(holder))

    # Completing the fit releases the lock ...
    files_path = Path(paths._files_path)
    files_path.mkdir(parents=True, exist_ok=True)
    (files_path / "samples_summary.json").write_text("{}")
    paths.completed()
    paths.zip_remove()
    assert not lock.exists()

    # ... and the finished archive replays read-only (no running lock).
    other.restore()
    assert not lock.exists() and other._archive_is_complete
    other.zip_remove()


def test__restore__dead_holder_lock_is_taken_over(tmp_path):
    import json
    import os

    paths = af.DirectoryPaths(name="dead_holder_test", path_prefix=str(tmp_path))
    paths.remove_files = True
    lock = paths._running_lock_path
    lock.parent.mkdir(parents=True, exist_ok=True)
    os.mkdir(lock)
    # A pid that cannot be alive on this host, no SLURM job: the holder is dead.
    (lock / "holder").write_text(json.dumps({"host": os.uname().nodename if hasattr(os, "uname") else "", "pid": 2**22 - 1}))

    paths.restore()
    assert paths._running_lock_held
    assert json.loads((lock / "holder").read_text())["pid"] == os.getpid()
    paths.zip_remove()
    assert not lock.exists()
