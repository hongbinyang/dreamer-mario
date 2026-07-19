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

The page is one long scroll, five sections:

- **Runs** — every run under `runs/` (Dreamer) and `runs_ppo/` (PPO baseline) in one table:
  name, step, best_x, flags, disk size, running/idle status. Row actions: **Dashboard** (opens
  TensorBoard for just that run), **Delete** (shells out to `cleanup.py --yes`, same as the CLI).
  Checkboxes here feed the Compare section below.
- **Start a run** — two side-by-side forms, **Start Dreamer training** and **Start PPO
  baseline**, covering the common flags (`total_steps`, `entropy_coef`/`ent_coef`,
  `sparse_reward`, device) directly, plus a free-text `--set KEY=VALUE` textarea (one override
  per line) for anything else in [configuration.md](configuration.md).
- **Evaluate & Dream** — pick a Dreamer run from a dropdown, then either **Evaluate**
  (episodes, optional video) or **Dream** (context/horizon/upscale/fps), mirroring
  [evaluation.md](evaluation.md). Output video lands under `runs/<name>/` alongside the
  checkpoint, and once the job finishes it plays back right there in the Active jobs list —
  see "Playing evaluation/dream videos" below.
- **Compare** — select 2+ runs (Dreamer and/or PPO, mixed freely) from the Runs table's
  checkboxes, click **Compare selected**. Opens a TensorBoard instance in a new tab, built from
  `dashboard.py --name ... --ppo-name ...` (`--logdir_spec` under the hood) — same mechanism
  [baselines.md](baselines.md#how-to-compare-after-both-have-run) documents for the CLI, so
  `episode/best_x`/`episode/flags` overlay directly on shared tag names.
- **Active jobs** — every tracked job (train, PPO, evaluate, dream, or a Compare's TensorBoard
  instance), whether started from the GUI or the CLI — jobs are tracked by PID, not by which
  process launched them. Shows the literal command line, a **Stop** button (sends the same
  `SIGINT` a terminal Ctrl-C would — see [training.md](training.md)'s "Ctrl-C any time" section),
  and an expandable live-tailing log view.

## Playing evaluation/dream videos

Evaluate (with "record video" checked) and Dream both write their output into the run's own
directory — `runs/<name>/eval_<job_id>.mp4`, `runs/<name>/dream_<job_id>.mp4` — the same place
`open_loop_<step>.gif` training snapshots already live. A Flask route,
`GET /files/<kind>/<name>/<filename>`, serves any file under a run's directory with HTTP Range
support (via `flask.send_from_directory`), which is what lets the browser's `<video>` element
actually scrub/seek instead of only playing from the start. Once a job's log shows its "wrote
..." completion line, the Active jobs entry renders a playable `<video controls>` pointed at that
route — no separate download step.

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
| Dashboard / Compare selected | `python scripts/dashboard.py [--name <dreamer-run>]... [--ppo-name <ppo-run>]... --port <ephemeral>` |

## How jobs are tracked

Every job the GUI launches (`subprocess.Popen(..., start_new_session=True)`, the same
detached-launch idiom as `nohup ... & disown`) gets a small JSON record under
`webui_state/jobs/<job_id>.json`: pid, the literal argv, name, kind, log path, start time.
Liveness is checked by PID (`os.kill(pid, 0)`, with proper zombie-reaping first), so the job list
survives a page reload or even restarting `scripts/webui.py` itself — it isn't in-memory-only
state. `webui_state/` is a new top-level directory, separate from `runs/`/`runs_ppo/` and
gitignored, so it's invisible to `cleanup.py`'s and `dashboard.py`'s existing directory scans —
starting the GUI has zero effect on those tools.

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
