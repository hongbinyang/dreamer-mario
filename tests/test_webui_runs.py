import sys
import time

import pytest
import torch
from torch.utils.tensorboard import SummaryWriter

from webui import runs


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


def _write_scalars(event_dir, best_x, flags):
    writer = SummaryWriter(str(event_dir))
    writer.add_scalar("episode/best_x", best_x, 1000)
    writer.add_scalar("episode/flags", flags, 1000)
    writer.close()


def test_scan_reads_a_dreamer_run(tmp_path):
    run_dir = tmp_path / "runs" / "trial"
    run_dir.mkdir(parents=True)
    torch.save({"step": 4200, "cfg": {}}, run_dir / "ckpt.pt")
    _write_scalars(run_dir, best_x=1500, flags=2)

    found = runs.scan()
    assert len(found) == 1
    r = found[0]
    assert r["name"] == "trial"
    assert r["kind"] == "dreamer"
    assert r["step"] == 4200
    assert r["best_x"] == 1500
    assert r["flags"] == 2
    # size_mb is rounded to 1 decimal for display (matches cleanup.py's
    # existing precision) -- these synthetic test files are too small
    # (~1.4KB) to show as nonzero there, so check the raw byte-level helper
    # directly instead of the rounded display field.
    assert runs._dir_size_mb(run_dir) > 0
    assert r["running"] is False


def test_scan_reads_a_ppo_run_from_its_nested_event_dir(tmp_path):
    # SB3 writes tfevents into a PPO_<n> subfolder, not the run dir itself --
    # this is the exact structure that broke a naive EventAccumulator(run_dir)
    # call earlier in this project (verified directly against real SB3 output).
    run_dir = tmp_path / "runs_ppo" / "ppo-trial"
    event_dir = run_dir / "PPO_1"
    event_dir.mkdir(parents=True)
    (run_dir / "model.zip").write_bytes(b"fake")
    _write_scalars(event_dir, best_x=900, flags=0)

    found = runs.scan()
    assert len(found) == 1
    r = found[0]
    assert r["name"] == "ppo-trial"
    assert r["kind"] == "ppo"
    assert r["step"] is None  # PPO has no agent.save()-style step counter
    assert r["best_x"] == 900
    assert r["flags"] == 0


def test_scan_picks_the_newest_ppo_event_subdir(tmp_path):
    run_dir = tmp_path / "runs_ppo" / "ppo-trial"
    (run_dir / "PPO_1").mkdir(parents=True)
    _write_scalars(run_dir / "PPO_1", best_x=100, flags=0)
    time.sleep(0.05)
    (run_dir / "PPO_2").mkdir(parents=True)
    _write_scalars(run_dir / "PPO_2", best_x=999, flags=1)

    found = runs.scan()
    assert found[0]["best_x"] == 999


def test_scan_run_with_no_checkpoint_yet_has_none_step(tmp_path):
    run_dir = tmp_path / "runs" / "just-started"
    run_dir.mkdir(parents=True)
    (run_dir / "train.log").write_text("device: mps\n")

    found = runs.scan()
    assert found[0]["step"] is None
    assert found[0]["best_x"] is None


def test_scan_reflects_a_live_tracked_job(tmp_path):
    run_dir = tmp_path / "runs" / "trial"
    run_dir.mkdir(parents=True)
    torch.save({"step": 0, "cfg": {}}, run_dir / "ckpt.pt")

    from webui import jobs
    record = jobs.launch([sys.executable, "-c", "import time; time.sleep(2)"],
                          name="trial", kind="train", log_path=tmp_path / "job.log")
    try:
        found = runs.scan()
        assert found[0]["running"] is True
    finally:
        jobs.stop(record["job_id"])


def test_scan_empty_when_no_run_directories_exist():
    assert runs.scan() == []


def test_scan_sorts_newest_first(tmp_path):
    old = tmp_path / "runs" / "old-run"
    old.mkdir(parents=True)
    torch.save({"step": 1}, old / "ckpt.pt")
    time.sleep(0.05)
    new = tmp_path / "runs" / "new-run"
    new.mkdir(parents=True)
    torch.save({"step": 1}, new / "ckpt.pt")

    found = runs.scan()
    assert [r["name"] for r in found] == ["new-run", "old-run"]


def test_scan_tolerates_a_corrupt_checkpoint_without_crashing(tmp_path):
    broken = tmp_path / "runs" / "broken-run"
    broken.mkdir(parents=True)
    (broken / "ckpt.pt").write_bytes(b"not a real checkpoint")

    found = runs.scan()
    assert found[0]["name"] == "broken-run"
    assert found[0]["step"] is None
