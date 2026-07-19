"""Flask app: thin HTTP glue over webui/runs.py and webui/jobs.py.

Every action shells out to the exact same CLI entry points a terminal user
would run -- this module never reimplements training/eval/dream/PPO/cleanup
logic, only builds argv lists and launches/tracks them. See docs/webui.md
for the full "GUI action -> CLI command" mapping.
"""
from __future__ import annotations

import pathlib
import socket
import subprocess
import sys

from flask import Flask, jsonify, render_template, request, send_from_directory

from . import jobs, runs

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
PYTHON = sys.executable
ROOTS = {"dreamer": "runs", "ppo": "runs_ppo"}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def _set_args(overrides: dict, extra_text: str) -> list[str]:
    """--set args from a {dotted.key: value} dict (skipping empty/None)
    plus a free-text textarea, one dotted.key=value override per line --
    covers the long tail of ~40+ config keys without a form field each."""
    args = []
    for key, value in overrides.items():
        if value is None or value == "":
            continue
        args += ["--set", f"{key}={value}"]
    for line in (extra_text or "").splitlines():
        line = line.strip()
        if line:
            args += ["--set", line]
    return args


def create_app() -> Flask:
    app = Flask(__name__)

    # ------------------------------------------------------------- page
    @app.get("/")
    def index():
        return render_template("index.html")

    # ------------------------------------------------------------- runs
    @app.get("/api/runs")
    def api_runs():
        return jsonify(runs.scan())

    @app.post("/api/runs/delete")
    def api_delete_run():
        data = request.get_json(force=True)
        name, kind = data["name"], data["kind"]
        result = subprocess.run(
            [PYTHON, "scripts/cleanup.py", "--logdir", ROOTS[kind], "--name", name, "--yes"],
            cwd=str(REPO_ROOT), capture_output=True, text=True)
        return jsonify({"ok": result.returncode == 0, "output": result.stdout + result.stderr})

    # ------------------------------------------------------------- jobs
    @app.get("/api/jobs")
    def api_jobs():
        return jsonify(jobs.list_jobs())

    @app.get("/api/jobs/<job_id>/log")
    def api_job_log(job_id):
        n = request.args.get("tail", 200, type=int)
        return jobs.tail_log(job_id, n), 200, {"Content-Type": "text/plain; charset=utf-8"}

    @app.post("/api/jobs/<job_id>/stop")
    def api_job_stop(job_id):
        return jsonify({"stopped": jobs.stop(job_id)})

    @app.post("/api/jobs/train")
    def api_start_train():
        data = request.get_json(force=True)
        name = data["name"]
        cmd = [PYTHON, "scripts/train.py", "--name", name]
        cmd += _set_args({
            "train.total_steps": data.get("total_steps"),
            "train.entropy_coef": data.get("entropy_coef"),
            "env.sparse_reward": "true" if data.get("sparse_reward") else None,
        }, data.get("set_text", ""))
        if data.get("device"):
            cmd += ["--device", data["device"]]
        log_path = REPO_ROOT / "runs" / name / "train.log"
        record = jobs.launch(cmd, name=name, kind="train", log_path=log_path, cwd=REPO_ROOT)
        return jsonify(record)

    @app.post("/api/jobs/ppo")
    def api_start_ppo():
        data = request.get_json(force=True)
        name = data["name"]
        cmd = [PYTHON, "baselines/ppo_baseline.py", "--name", name]
        cmd += _set_args({
            "ppo.total_steps": data.get("total_steps"),
            "ppo.ent_coef": data.get("ent_coef"),
            "env.sparse_reward": "true" if data.get("sparse_reward") else None,
        }, data.get("set_text", ""))
        if data.get("device"):
            cmd += ["--device", data["device"]]
        log_path = REPO_ROOT / "runs_ppo" / name / "train.log"
        record = jobs.launch(cmd, name=name, kind="ppo", log_path=log_path, cwd=REPO_ROOT)
        return jsonify(record)

    @app.post("/api/jobs/evaluate")
    def api_start_evaluate():
        data = request.get_json(force=True)
        name = data["name"]
        job_id = jobs.new_job_id()
        video_filename = f"eval_{job_id}.mp4" if data.get("video") else None
        cmd = [PYTHON, "scripts/evaluate.py", "--name", name,
               "--episodes", str(data.get("episodes", 5))]
        if video_filename:
            cmd += ["--video", f"runs/{name}/{video_filename}"]
        cmd += _set_args({}, data.get("set_text", ""))
        log_path = REPO_ROOT / "runs" / name / f"eval_{job_id}.log"
        record = jobs.launch(cmd, name=name, kind="evaluate", log_path=log_path,
                              cwd=REPO_ROOT, job_id=job_id)
        record["video_filename"] = video_filename
        return jsonify(record)

    @app.post("/api/jobs/dream")
    def api_start_dream():
        data = request.get_json(force=True)
        name = data["name"]
        job_id = jobs.new_job_id()
        out_filename = f"dream_{job_id}.mp4"
        cmd = [PYTHON, "scripts/dream.py", "--name", name, "--out", f"runs/{name}/{out_filename}",
               "--context", str(data.get("context", 8)), "--horizon", str(data.get("horizon", 56)),
               "--upscale", str(data.get("upscale", 4))]
        if data.get("fps"):
            cmd += ["--fps", str(data["fps"])]
        cmd += _set_args({}, data.get("set_text", ""))
        log_path = REPO_ROOT / "runs" / name / f"dream_{job_id}.log"
        record = jobs.launch(cmd, name=name, kind="dream", log_path=log_path,
                              cwd=REPO_ROOT, job_id=job_id)
        record["video_filename"] = out_filename
        return jsonify(record)

    # ---------------------------------------------------------- compare
    @app.post("/api/compare")
    def api_compare():
        data = request.get_json(force=True)
        entries = data["runs"]  # [{"name": ..., "kind": "dreamer"|"ppo"}, ...]
        port = _free_port()
        cmd = [PYTHON, "scripts/dashboard.py", "--port", str(port)]
        for e in entries:
            cmd += ["--ppo-name" if e["kind"] == "ppo" else "--name", e["name"]]
        name_label = "+".join(e["name"] for e in entries)
        log_path = REPO_ROOT / "webui_state" / "dashboards" / f"{port}.log"
        record = jobs.launch(cmd, name=name_label, kind="dashboard", log_path=log_path,
                              cwd=REPO_ROOT)
        record["url"] = f"http://localhost:{port}/"
        return jsonify(record)

    # ---------------------------------------------------------- files
    @app.get("/files/<kind>/<name>/<path:filename>")
    def files(kind, name, filename):
        if kind not in ROOTS:
            return "unknown kind", 404
        # send_from_directory rejects path-traversal attempts (leading '..'
        # or absolute paths) on its own; no extra checking needed here.
        return send_from_directory(str(REPO_ROOT / ROOTS[kind] / name), filename)

    return app
