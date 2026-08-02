"""The ingest endpoint's front door.

Only one thing here is worth testing hard, and it is not the HTTP plumbing. This endpoint takes a
filesystem path chosen by a browser and runs a program on it, so the containment check is the
whole security boundary and every way around it belongs in this file.
"""

from __future__ import annotations

import pytest

from verdict.serve import Runner


@pytest.fixture
def root(tmp_path):
    """A permitted root holding one valid release, with a secret parked outside it."""
    data = tmp_path / "data"
    release = data / "unseen"
    release.mkdir(parents=True)
    (release / "ad_events.parquet").write_bytes(b"parquet")

    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "ad_events.parquet").write_bytes(b"not yours")
    return data


@pytest.fixture
def runner(root):
    return Runner(root)


class TestItAcceptsARelease:
    def test_a_directory_holding_events(self, runner, root):
        assert runner.resolve(str(root / "unseen")) == (root / "unseen").resolve()

    def test_the_parquet_itself(self, runner, root):
        target = root / "unseen" / "ad_events.parquet"
        assert runner.resolve(str(target)) == target.resolve()

    def test_a_path_relative_to_the_root(self, runner, root):
        """Typing the release name rather than its full path is the common case."""
        assert runner.resolve("unseen") == (root / "unseen").resolve()


class TestItRefusesEverythingElse:
    def test_an_empty_path(self, runner):
        with pytest.raises(ValueError, match="required"):
            runner.resolve("   ")

    def test_a_path_that_does_not_exist(self, runner, root):
        with pytest.raises(ValueError, match="No such path"):
            runner.resolve(str(root / "nope"))

    def test_a_sibling_directory_outside_the_root(self, runner, root):
        with pytest.raises(PermissionError, match="outside the permitted root"):
            runner.resolve(str(root.parent / "elsewhere"))

    def test_a_traversal_out_of_the_root(self, runner):
        with pytest.raises(PermissionError, match="outside the permitted root"):
            runner.resolve("../elsewhere")

    def test_a_symlink_pointing_out_of_the_root(self, runner, root):
        """The reason the check resolves before comparing. A link whose name sits inside the root
        and whose target does not is the standard way past a string-prefix test."""
        link = root / "sneaky"
        link.symlink_to(root.parent / "elsewhere")
        with pytest.raises(PermissionError, match="outside the permitted root"):
            runner.resolve(str(link))

    def test_a_directory_with_no_events_in_it(self, runner, root):
        empty = root / "empty"
        empty.mkdir()
        with pytest.raises(ValueError, match="ad_events.parquet"):
            runner.resolve(str(empty))

    def test_a_file_that_is_not_a_parquet(self, runner, root):
        stray = root / "notes.txt"
        stray.write_text("hello")
        with pytest.raises(ValueError, match="directory or a .parquet"):
            runner.resolve(str(stray))


class TestItRunsOneJobAtATime:
    def test_a_second_job_is_refused_while_one_holds_the_lock(self, runner, root):
        """Two concurrent ingests would interleave inserts into the same table and leave the
        rollups describing neither batch."""
        target = (root / "unseen").resolve()
        runner._lock.acquire()
        try:
            with pytest.raises(RuntimeError, match="already running"):
                runner.run(target)
        finally:
            runner._lock.release()

    def test_the_lock_is_released_after_a_run(self, runner, root, monkeypatch):
        import subprocess

        class Done:
            returncode = 0
            stdout = "ingested"
            stderr = ""

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: Done())
        target = (root / "unseen").resolve()
        assert runner.run(target)["ok"] is True
        assert runner.busy is False
        # A second call must still be able to take the lock.
        assert runner.run(target)["ok"] is True

    def test_a_failing_command_is_reported_not_raised(self, runner, root, monkeypatch):
        import subprocess

        class Failed:
            returncode = 2
            stdout = "partial output"
            stderr = "boom"

        monkeypatch.setattr(subprocess, "run", lambda *a, **k: Failed())
        result = runner.run((root / "unseen").resolve())
        assert result["ok"] is False
        assert result["exitCode"] == 2
        assert "boom" in result["output"]
