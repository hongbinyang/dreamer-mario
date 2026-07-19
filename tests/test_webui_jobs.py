import sys
import time

import pytest

from webui import jobs


@pytest.fixture(autouse=True)
def isolated_registry(tmp_path, monkeypatch):
    # jobs.py's registry path (webui_state/jobs/) is a plain relative path,
    # same convention as runs/ elsewhere in this project -- redirect it into
    # a throwaway directory for the whole test rather than touching the
    # real repo's webui_state/.
    monkeypatch.chdir(tmp_path)


def _wait_until_dead(job_id, timeout=3.0):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not jobs.get_job(job_id)["alive"]:
            return True
        time.sleep(0.05)
    return False


def test_launch_tracks_a_real_process_and_stop_sends_sigint(tmp_path):
    log_path = tmp_path / "run" / "job.log"
    record = jobs.launch(
        [sys.executable, "-c", "import time; time.sleep(5)"],
        name="test-run", kind="train", log_path=log_path)
    assert record["pid"] > 0
    assert record["name"] == "test-run"

    job = jobs.get_job(record["job_id"])
    assert job["alive"] is True
    assert jobs.is_run_active("test-run") is True

    assert jobs.stop(record["job_id"]) is True
    assert _wait_until_dead(record["job_id"])
    # Not just a one-time transition -- repeated checks after exit must
    # stay stable (this is exactly the zombie-reaping bug: without proper
    # waitpid() reaping, os.kill(pid, 0) would keep reporting "alive").
    assert jobs.get_job(record["job_id"])["alive"] is False
    assert jobs.get_job(record["job_id"])["alive"] is False
    assert jobs.is_run_active("test-run") is False


def test_stop_on_already_finished_job_returns_false(tmp_path):
    log_path = tmp_path / "run" / "job.log"
    record = jobs.launch([sys.executable, "-c", "pass"], name="x", kind="dream", log_path=log_path)
    assert _wait_until_dead(record["job_id"])
    assert jobs.stop(record["job_id"]) is False


def test_launch_writes_subprocess_output_to_the_log_file(tmp_path):
    log_path = tmp_path / "run" / "job.log"
    record = jobs.launch(
        [sys.executable, "-c", "print('hello from job')"],
        name="test-run", kind="evaluate", log_path=log_path)
    assert _wait_until_dead(record["job_id"])
    assert "hello from job" in jobs.tail_log(record["job_id"])


def test_launch_respects_explicit_cwd(tmp_path):
    workdir = tmp_path / "somewhere-else"
    workdir.mkdir()
    log_path = tmp_path / "job.log"
    record = jobs.launch(
        [sys.executable, "-c", "import pathlib; print(pathlib.Path.cwd())"],
        name="test-run", kind="evaluate", log_path=log_path, cwd=workdir)
    assert _wait_until_dead(record["job_id"])
    assert str(workdir.resolve()) in jobs.tail_log(record["job_id"])


def test_list_jobs_returns_newest_first(tmp_path):
    r1 = jobs.launch([sys.executable, "-c", "pass"], name="a", kind="train",
                      log_path=tmp_path / "a.log")
    time.sleep(0.05)
    r2 = jobs.launch([sys.executable, "-c", "pass"], name="b", kind="train",
                      log_path=tmp_path / "b.log")
    ids = [j["job_id"] for j in jobs.list_jobs()]
    assert ids[0] == r2["job_id"]
    assert r1["job_id"] in ids


def test_get_job_unknown_id_returns_none():
    assert jobs.get_job("does-not-exist") is None


def test_launch_accepts_a_pregenerated_job_id(tmp_path):
    # Needed so callers can embed the id in an output filename (e.g.
    # evaluate's --video eval_<job_id>.mp4) before the job exists.
    job_id = jobs.new_job_id()
    record = jobs.launch([sys.executable, "-c", "pass"], name="x", kind="evaluate",
                          log_path=tmp_path / "job.log", job_id=job_id)
    assert record["job_id"] == job_id
    assert jobs.get_job(job_id) is not None


def test_is_run_active_false_when_no_jobs_for_that_name(tmp_path):
    record = jobs.launch([sys.executable, "-c", "import time; time.sleep(1)"],
                          name="other-run", kind="train", log_path=tmp_path / "job.log")
    assert jobs.is_run_active("nonexistent-run") is False
    assert _wait_until_dead(record["job_id"])  # don't leave it running past the test
