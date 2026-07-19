"""Regression test for a real bug found while building the web GUI: a shell
backgrounding a process (`cmd &` -- including the nohup/disown recipe
docs/training.md itself recommends, and how scripts/webui.py always launches
jobs) sets SIGINT to SIG_IGN in the child *before* exec, to protect
background jobs from a terminal's Ctrl-C. SIG_IGN survives exec(), and
CPython's interpreter startup deliberately leaves an already-ignored SIGINT
alone rather than installing its own handler -- so without an explicit
reset, a backgrounded train.py is *permanently* immune to SIGINT, silently
breaking the "Ctrl-C any time is safe" guarantee documented everywhere.

Verified directly against the real bug (not just this test) with the exact
nohup+disown command from training.md's docstring plus `kill -INT` from a
separate shell: the process never stopped without the fix, and stopped in
~1s with it.

preexec_fn here deterministically reproduces exactly the "inherited
SIG_IGN" condition a shell's `cmd &` produces, without depending on actual
shell job-control behavior (which would make this test flaky/environment-
dependent).
"""
from __future__ import annotations

import pathlib
import shutil
import signal
import subprocess
import sys
import time

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
RUN_NAME = "_sigtest_regression"


def _ignore_sigint():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


@pytest.fixture(autouse=True)
def cleanup_run_dir():
    yield
    shutil.rmtree(REPO_ROOT / "runs" / RUN_NAME, ignore_errors=True)


def test_train_overrides_an_inherited_sigint_ignore(tmp_path):
    cmd = [sys.executable, "scripts/train.py", "--name", RUN_NAME, "--device", "cpu",
           "--set", "model.cnn_depth=8", "--set", "model.deter=16", "--set", "model.stoch=4",
           "--set", "model.classes=4", "--set", "model.hidden=16", "--set", "replay.batch_size=4",
           "--set", "replay.seq_len=8", "--set", "replay.prefill=20", "--set", "replay.capacity=200",
           "--set", "train.imag_horizon=3", "--set", "train.total_steps=1000000",
           "--set", "run.checkpoint_every=1000000", "--set", "run.video_every=1000000"]
    log_path = tmp_path / "train.log"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 cwd=str(REPO_ROOT), preexec_fn=_ignore_sigint)
    try:
        time.sleep(3)  # let it get past prefill/argparse into the real training loop
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            pytest.fail(
                "train.py did not respond to SIGINT within 8s when launched with an "
                "inherited SIG_IGN disposition -- this reproduces the backgrounded-shell "
                "bug; see this file's module docstring")
    finally:
        if proc.poll() is None:
            proc.kill()


def test_dashboard_overrides_an_inherited_sigint_ignore_and_kills_tensorboard_child(tmp_path):
    """Found while testing the web GUI's Compare 'Stop' button: dashboard.py
    was missed from the original SIGINT-reset fix (it isn't one of
    train.py/evaluate.py/dream.py/ppo_baseline.py), so a dashboard job
    launched by the GUI (itself typically backgrounded) was permanently
    immune to Stop -- and even after resetting SIGINT the same way, the
    *tensorboard* process dashboard.py spawns via a blocking subprocess.run()
    call needed checking too, since killing the wrapper alone wouldn't be
    enough. Verified directly: subprocess.run()'s own kill()-on-exception
    behavior (documented in dashboard.py's comment) already takes care of
    the child once the KeyboardInterrupt this reset enables actually fires."""
    run_dir = tmp_path / "runs" / "trial"
    run_dir.mkdir(parents=True)
    port = 61987
    cmd = [sys.executable, "scripts/dashboard.py", "--name", "trial",
           "--logdir", str(tmp_path / "runs"), "--port", str(port)]
    log_path = tmp_path / "dashboard.log"
    with open(log_path, "wb") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                 cwd=str(REPO_ROOT), preexec_fn=_ignore_sigint)
    try:
        time.sleep(2)  # let tensorboard actually bind and start serving
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=8)
        except subprocess.TimeoutExpired:
            pytest.fail(
                "dashboard.py did not respond to SIGINT within 8s when launched with an "
                "inherited SIG_IGN disposition -- see this test's docstring")
        time.sleep(0.5)
        still_listening = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True
        ).stdout
        assert f"--port {port}" not in still_listening, (
            "dashboard.py exited but its tensorboard child is still running -- "
            "Stop must kill both, not just the wrapper")
    finally:
        if proc.poll() is None:
            proc.kill()
