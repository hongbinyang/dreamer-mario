"""Launch the web GUI: start/stop/monitor/evaluate/compare runs from a browser.

    pip install flask   # optional, not in requirements
    python scripts/webui.py
    python scripts/webui.py --port 9000

Every action in the GUI shells out to the exact same CLI entry points this
file's siblings expose -- see docs/webui.md for the full page walkthrough
and a "GUI action -> CLI command" table.

Binds to localhost only by default. See docs/webui.md before ever passing
--host 0.0.0.0: there is no authentication, and the GUI can execute
training runs and delete files.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from dreamer.config import dict_to_ns, load_yaml  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", action="append", default=[],
                         help="dotted.key=value overrides, e.g. --set webui.port=9000")
    parser.add_argument("--host", default=None, help="overrides webui.host (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=None, help="overrides webui.port (default 8000)")
    args = parser.parse_args()

    cfg = dict_to_ns(load_yaml(args.config, args.set))
    host = args.host or cfg.webui.host
    port = args.port or cfg.webui.port

    from webui.app import create_app
    app = create_app()
    print(f"dreamer-mario web GUI: http://{host}:{port}")
    # use_reloader=False always -- Flask's reloader spawns a second process,
    # which would confuse our own subprocess/pidfile job tracking.
    app.run(host=host, port=port, use_reloader=False)


if __name__ == "__main__":
    main()
