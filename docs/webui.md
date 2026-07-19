# Web GUI

A browser front end over the same CLI scripts documented elsewhere in this project — nothing
in `webui/` reimplements training/evaluation/dream/PPO/cleanup logic. Every button click shells
out to the exact same command a terminal user would run, so a run started from the browser is
indistinguishable from one started with `python scripts/train.py --name ...` directly: you can
start it in the GUI, then `tail`/resume/delete it from a terminal, or vice versa, with no
divergence. This page has no effect on any existing CLI workflow — it's a layer on top, not a
replacement.

## Setup

```bash
pip install flask   # optional, not in requirements.txt/environment.yml
python scripts/webui.py
python scripts/webui.py --port 9000
```

Then open `http://127.0.0.1:8000` (or whatever `--port` you passed).

**Binds to `127.0.0.1` only by default.** There is no authentication, and the GUI can execute
training runs and delete run directories — think twice before ever passing `--host 0.0.0.0`
(exposes it to your whole network) or `--set webui.host=0.0.0.0` in `configs/default.yaml`.

## Page walkthrough

The page is one long scroll. Every action panel shows its own status inline, right below its
form — nothing requires scrolling elsewhere to see what's happening.

- **Runs** — every run under `runs/` (Dreamer) and `runs_ppo/` (PPO baseline) in one table:
  name, step, best_x, flags, disk size, running/idle status. Row actions: **Dashboard** (opens
  TensorBoard for just that run), **Delete** (shells out to `cleanup.py --yes`, same as the CLI).
  Checkboxes here feed the Compare section below.
- **Start a run** — two side-by-side forms, **Start Dreamer training** and **Start PPO
  baseline**, covering the common flags (`total_steps`, `entropy_coef`/`ent_coef`,
  `sparse_reward`, device) directly, plus a free-text `--set KEY=VALUE` textarea (one override
  per line) for anything else in [configuration.md](configuration.md). A **training jobs** list
  sits directly under this section — every `train`/`ppo` job the GUI has launched, across every
  run name at once, each with its command line, a **Stop** button, and an expandable log.
- **Evaluate & Dream** — pick a Dreamer run from a dropdown to target a *new* job, then either
  **Evaluate** (episodes, optional video) or **Dream** (context/horizon/upscale/fps), mirroring
  [evaluation.md](evaluation.md). Below each form is that panel's own history — every
  evaluate/dream job across *every* run name, not just whichever one is currently selected in the
  dropdown above it (switching the dropdown only changes what a new submission targets, not what
  the history shows). See "Evaluate/Dream lifecycle" below for what each entry offers.
- **Compare** — select 2+ runs (Dreamer and/or PPO, mixed freely) from the Runs table's
  checkboxes, click **Compare selected**. Opens a TensorBoard instance in a new tab, built from
  `dashboard.py --name ... --ppo-name ...` (`--logdir_spec` under the hood) — same mechanism
  [baselines.md](baselines.md#how-to-compare-after-both-have-run) documents for the CLI, so
  `episode/best_x`/`episode/flags` overlay directly on shared tag names. An **active dashboards**
  list appears inline below the button once anything is running — see "Comparing twice" below.

## Evaluate/Dream lifecycle: status, logs, delete, multiple clicks

Each row in the Evaluate/Dream history is one job, labeled by run name, and reflects the same
disk-first philosophy as the Runs table: it shows up whether it was started from the GUI or from
a terminal, and disappears the moment its files are gone, whether deleted through the GUI or with
`rm`.

- **Status**: `running` while the underlying `evaluate.py`/`dream.py` process is alive; `finished`
  once it exits; `finished (started outside the GUI)` for a video file the GUI found on disk with
  no matching job record — e.g. one produced by running `evaluate.py`/`dream.py` directly from a
  terminal with a `runs/<name>/eval_*.mp4`/`dream_*.mp4`-style `--video`/`--out` path. An orphan
  entry like this still gets a log view if a same-stem `.log` file happens to sit next to it, but
  never a Stop button — there's no PID to send `SIGINT` to for a process this GUI didn't launch.
- **Logs**: every entry with a known log gets a **Toggle log** button, expanding the same
  live-tailing view line-for-line as `evaluate.py`/`dream.py` would print to a terminal.
- **Delete**: once finished, a **Delete** button removes the video and its log (mirrors the Runs
  table's whole-run Delete, just scoped to one evaluate/dream output instead of an entire run).
  Only ever offered once finished — a running job hasn't written its video yet (`evaluate.py`/
  `dream.py` both write the file once, at the very end, not incrementally), so there's no
  running/deleting overlap to worry about.
- **Multiple clicks**: not blocked, and deliberately so — each click is a fresh job reading a
  checkpoint it never modifies and writing to its own uniquely-named output file, so there's no
  data race the way there would be for two `train.py` invocations sharing one `--name`. Click
  Evaluate three times and you get three rows, each independently tracked; that's the point of the
  inline history existing at all, rather than trying to guess which single click you meant.

## Comparing twice

Clicking **Compare selected** again with the *exact same* run selection re-opens the existing
TensorBoard tab instead of launching a second `tensorboard` process on a second port — the backend
checks already-running dashboard jobs for an identical run-set before spawning a new one. A
*different* selection is a legitimately new comparison and gets its own process. Once you `Stop` a
dashboard entry (or it's never been started), a matching selection launches fresh again.

`scripts/dashboard.py` needed the same `SIGINT`-reset fix `docs/training.md` documents for
`train.py` (a GUI-launched job is typically backgrounded, which otherwise leaves `SIGINT`
permanently ignored) — without it, a dashboard job's Stop button would appear to do nothing.
Fixing just the wrapper wasn't quite enough on its own, though: `dashboard.py` launches the actual
`tensorboard` process via a blocking `subprocess.run()` call, so it also needed confirming that
the real `tensorboard` server dies along with it, not just the wrapper script — verified directly,
`subprocess.run()`'s own behavior of killing its child on any exception (including the
`KeyboardInterrupt` the reset enables) already takes care of that.

## Playing evaluation/dream videos

Evaluate (with "record video" checked) and Dream both write their output into the run's own
directory — `runs/<name>/eval_<job_id>.mp4`, `runs/<name>/dream_<job_id>.mp4` — the same place
`open_loop_<step>.gif` training snapshots already live. A Flask route,
`GET /files/<kind>/<name>/<filename>`, serves any file under a run's directory with HTTP Range
support (via `flask.send_from_directory`), which is what lets the browser's `<video>` element
actually scrub/seek instead of only playing from the start. Once a job finishes, its history entry
renders a playable `<video controls>` pointed at that route — no separate download step.

## Restart survival

Closing the browser tab, or even fully killing and relaunching `scripts/webui.py` itself, doesn't
lose track of anything. Nothing about job tracking lives in memory: liveness is a fresh
`os.kill(pid, 0)` check on every request, and detached subprocesses
(`start_new_session=True`, the same idiom as `nohup ... & disown`) aren't tied to the launching
Flask process's lifetime — they get reparented and keep running if it dies. Reopening the page (or
starting a brand-new `scripts/webui.py` process) just re-reads the same `webui_state/jobs/*.json`
registry and re-checks the same PIDs; a still-running training job picks right back up as
`running` with a working Stop button, and finished evaluate/dream artifacts are found by disk scan
regardless of registry state.

## GUI action → CLI command

For full transparency (and so you can copy an action into a terminal, e.g. to add flags the GUI
doesn't expose a field for):

| GUI action | Equivalent CLI command |
|---|---|
| Start Dreamer training | `python scripts/train.py --name <name> [--set train.total_steps=... --set train.entropy_coef=... --set env.sparse_reward=true] [--device ...] [--set ...]` |
| Start PPO baseline | `python baselines/ppo_baseline.py --name <name> [--set ppo.total_steps=... --set ppo.ent_coef=... --set env.sparse_reward=true] [--device ...] [--set ...]` |
| Evaluate | `python scripts/evaluate.py --name <name> --episodes <n> [--video runs/<name>/eval_<job_id>.mp4] [--set ...]` |
| Dream | `python scripts/dream.py --name <name> --out runs/<name>/dream_<job_id>.mp4 --context <c> --horizon <h> --upscale <u> [--fps ...] [--set ...]` |
| Stop (any job) | `kill -INT <pid>` — same signal a terminal Ctrl-C sends |
| Delete (a run row) | `python scripts/cleanup.py --logdir <runs|runs_ppo> --name <name> --yes` |
| Delete (an evaluate/dream history row) | `rm runs/<name>/<eval\|dream>_<id>.mp4 runs/<name>/<eval\|dream>_<id>.log` |
| Dashboard / Compare selected | `python scripts/dashboard.py [--name <dreamer-run>]... [--ppo-name <ppo-run>]... --port <ephemeral>` |

## How jobs are tracked

Every job the GUI launches (`subprocess.Popen(..., start_new_session=True)`, the same
detached-launch idiom as `nohup ... & disown`) gets a small JSON record under
`webui_state/jobs/<job_id>.json`: pid, the literal argv, name, kind, log path, start time.
Liveness is checked by PID (`os.kill(pid, 0)`, with proper zombie-reaping first) — see "Restart
survival" above for what this buys you. `webui_state/` is a new top-level directory, separate from
`runs/`/`runs_ppo/` and gitignored, so it's invisible to `cleanup.py`'s and `dashboard.py`'s
existing directory scans — starting the GUI has zero effect on those tools. Evaluate/dream
artifacts (the videos themselves) are *not* tracked here at all — they're discovered by scanning
each run's directory directly, which is what lets a terminal-produced video show up in the
Evaluate/Dream history without ever touching this registry.

## `scripts/webui.py` options

| Flag | Default | Meaning |
|---|---|---|
| `--config` | `configs/default.yaml` | YAML config to read the `webui:` defaults from. |
| `--set KEY=VALUE` | *(none; repeatable)* | Dotted-key override, e.g. `--set webui.port=9000` — same mechanism as every other script. |
| `--host` | `None` (→ `webui.host`, i.e. `127.0.0.1`) | Interface to bind. See the security note above before changing this. |
| `--port` | `None` (→ `webui.port`, i.e. `8000`) | Port to bind. |

## `webui` config keys (`configs/default.yaml`)

| Key | Default | Meaning |
|---|---|---|
| `host` | `127.0.0.1` | Default bind address for `scripts/webui.py`. |
| `port` | `8000` | Default port for `scripts/webui.py`. |
