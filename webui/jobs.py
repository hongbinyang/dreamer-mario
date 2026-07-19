"""Launch, track, and stop background CLI subprocesses on behalf of the web
GUI. No knowledge of what the subprocess actually does -- callers pass a
fully-formed argv (e.g. [sys.executable, "scripts/train.py", "--name", ...]),
this module just launches it detached and tracks it.

Job metadata (pid, argv, log path) lives under webui_state/jobs/<job_id>.json
-- deliberately separate from runs/ and runs_ppo/, so it's invisible to
scripts/cleanup.py's and scripts/dashboard.py's existing directory scans.
Liveness is checked live via os.kill(pid, 0) rather than trusted from the
file, so tracking survives a webui restart or a job that was killed outside
the GUI.
"""
from __future__ import annotations

import json
import os
import pathlib
import signal
import subprocess
import time
import uuid


def _registry_dir() -> pathlib.Path:
    d = pathlib.Path("webui_state") / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _registry_path(job_id: str) -> pathlib.Path:
    return _registry_dir() / f"{job_id}.json"


def _is_alive(pid: int) -> bool:
    # Reap the process first if it has already exited. We're its parent (no
    # double-fork daemonization here, just start_new_session for nohup/
    # disown-equivalent detachment), so an exited-but-unreaped child is a
    # zombie -- os.kill(pid, 0) would keep reporting it "alive" forever
    # otherwise, since our own process stays running for the whole GUI
    # session and the OS never reaps our children on our behalf.
    try:
        reaped_pid, _ = os.waitpid(pid, os.WNOHANG)
        if reaped_pid == pid:
            return False
    except ChildProcessError:
        pass  # not our child (already reaped earlier, or never was)
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def new_job_id() -> str:
    """Exposed so callers that need to embed the job id in an output
    filename (e.g. evaluate's --video eval_<job_id>.mp4) can generate one
    before building the command, then pass it to launch() below."""
    return f"{int(time.time())}-{uuid.uuid4().hex[:8]}"


def launch(cmd: list[str], *, name: str, kind: str, log_path: pathlib.Path,
           cwd: pathlib.Path | None = None, job_id: str | None = None,
           extra: dict | None = None) -> dict:
    """Starts cmd detached (nohup/disown-equivalent), stdout+stderr to
    log_path. cwd matters: train.py/evaluate.py/etc. resolve --config and
    similar relative paths against their runtime working directory, not
    their own file location, so callers should pass the repo root
    explicitly rather than relying on whatever CWD this process happens to
    have. extra is merged into the saved record as-is -- e.g. the dashboard
    job kind stores {"entries": [...]} there to de-duplicate future Compare
    requests against the exact same run-set. Returns the job record (see
    list_jobs)."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "wb") as log_file:
        proc = subprocess.Popen(
            cmd, stdout=log_file, stderr=subprocess.STDOUT, start_new_session=True,
            cwd=str(cwd) if cwd else None)

    job_id = job_id or new_job_id()
    record = {
        "job_id": job_id,
        "pid": proc.pid,
        "cmd": cmd,
        "name": name,
        "kind": kind,  # "train" | "ppo" | "evaluate" | "dream" | "dashboard"
        "log_path": str(log_path),
        "started_at": time.time(),
        **(extra or {}),
    }
    _registry_path(job_id).write_text(json.dumps(record))
    return record


def list_jobs() -> list[dict]:
    """All tracked jobs (any status), newest first, each with a live 'alive' flag."""
    out = []
    for p in sorted(_registry_dir().glob("*.json")):
        try:
            record = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        record["alive"] = _is_alive(record["pid"])
        out.append(record)
    return sorted(out, key=lambda r: r["started_at"], reverse=True)


def get_job(job_id: str) -> dict | None:
    p = _registry_path(job_id)
    if not p.exists():
        return None
    record = json.loads(p.read_text())
    record["alive"] = _is_alive(record["pid"])
    return record


def stop(job_id: str) -> bool:
    """Sends SIGINT (the same signal a terminal Ctrl-C sends) -- not
    SIGKILL, so it hits the same safe-stop path (atomic checkpoint writes,
    clean resume) documented for a terminal Ctrl-C."""
    record = get_job(job_id)
    if not record or not record["alive"]:
        return False
    os.kill(record["pid"], signal.SIGINT)
    return True


def delete_job_record(job_id: str) -> bool:
    """Removes a job's registry entry only -- never touches the process
    (call stop() first if it might still be alive) or any output files it
    produced. Used when deleting a finished evaluate/dream artifact, so the
    now-pointless registry entry doesn't linger and get re-surfaced as an
    "orphan" the next time its (now-deleted) video is looked for."""
    p = _registry_path(job_id)
    if not p.exists():
        return False
    p.unlink()
    return True


def is_run_active(name: str) -> bool:
    """Whether any tracked job for this run name is currently alive.
    Only reflects jobs launched through this GUI -- a run started directly
    from the terminal won't show as active here (see docs/webui.md)."""
    return any(j["name"] == name and j["alive"] for j in list_jobs())


def tail_log(job_id: str, n: int = 200) -> str:
    record = get_job(job_id)
    if not record:
        return ""
    path = pathlib.Path(record["log_path"])
    if not path.exists():
        return ""
    lines = path.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])
