#!/usr/bin/env python3
"""One page that follows the Camões LSTM training log.

On this VPS it pulls the log from RunPod over SSH. On the training
machine, leave CAMOES_SSH empty and it reads the log file directly.

    cp follow.env.example follow.env
    python3 follow.py
"""

import json
import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def load_env(path: Path) -> dict:
    vals = {}
    if not path.is_file():
        return vals
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        vals[key.strip()] = value.strip().strip('"').strip("'")
    return vals


def config() -> dict:
    file_vals = load_env(ROOT / "follow.env")

    def pick(name, default=""):
        return os.environ.get(name, file_vals.get(name, default))

    key = pick("CAMOES_SSH_KEY", "~/.ssh/id_ed25519")
    return {
        "ssh": pick("CAMOES_SSH"),
        "key": str(Path(key).expanduser()),
        "log": pick("CAMOES_LOG", "/workspace/camoes/train.log"),
        "bind": pick("BIND", "0.0.0.0"),
        "port": int(pick("PORT", "8090")),
        "interval": float(pick("POLL_SECONDS", "3")),
    }


def parse_log(text: str) -> dict:
    text = ANSI.sub("", text)
    epochs = []
    sample = ""
    sample_epoch = None
    in_sample = False
    buf = []
    step = total = None
    loss = rate = None
    done = None
    header = []
    for line in text.splitlines():
        if in_sample:
            if line.startswith("--- end sample ---"):
                in_sample = False
                sample = "\n".join(buf).strip("\n")
            else:
                buf.append(line)
            continue
        if line.startswith("--- sample epoch "):
            parts = line.split()
            sample_epoch = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else None
            in_sample = True
            buf = []
            continue
        matched = re.match(
            r"step (\d+)/(\d+)\s+loss=([0-9.]+)\s+([0-9.]+) step/s", line
        )
        if matched:
            step = int(matched.group(1))
            total = int(matched.group(2))
            loss = float(matched.group(3))
            rate = float(matched.group(4))
            continue
        matched = re.match(
            r"epoch (\d+)\s+train=([0-9.]+)\s+val=([0-9.]+)\s+ppl=([0-9.]+)",
            line,
        )
        if matched:
            epochs.append(
                {
                    "epoch": int(matched.group(1)),
                    "train": float(matched.group(2)),
                    "val": float(matched.group(3)),
                    "ppl": float(matched.group(4)),
                }
            )
            continue
        matched = re.match(
            r"done steps=(\d+) last_loss=([0-9.]+) elapsed_min=([0-9.]+)", line
        )
        if matched:
            done = {
                "steps": int(matched.group(1)),
                "last_loss": float(matched.group(2)),
                "elapsed_min": float(matched.group(3)),
            }
            continue
        if line.startswith(("gpu=", "chars=", "hidden=", "parameters=", "Downloading")):
            header.append(line)

    if done:
        phase = "done"
    elif step:
        phase = "running"
    elif text.strip():
        phase = "starting"
    else:
        phase = "idle"

    best = min(epochs, key=lambda row: row["val"]) if epochs else None
    last = epochs[-1] if epochs else None
    return {
        "phase": phase,
        "step": step,
        "total_steps": total,
        "loss": loss,
        "steps_per_s": rate,
        "epoch": last["epoch"] if last else None,
        "train": last["train"] if last else None,
        "val": last["val"] if last else None,
        "ppl": last["ppl"] if last else None,
        "best_epoch": best["epoch"] if best else None,
        "best_val": best["val"] if best else None,
        "elapsed_min": done["elapsed_min"] if done else None,
        "epochs": epochs,
        "sample_epoch": sample_epoch,
        "sample": sample,
        "header": header,
    }


def fetch_remote(cfg: dict) -> str:
    script = (
        "echo ___LOG_START___\n"
        f"cat {cfg['log']}\n"
        "echo ___LOG_END___\n"
        "exit\n"
    )
    proc = subprocess.run(
        [
            "ssh",
            "-tt",
            "-o",
            "ConnectTimeout=20",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-i",
            cfg["key"],
            cfg["ssh"],
            "bash",
            "-s",
        ],
        input=script.encode(),
        capture_output=True,
        timeout=45,
    )
    out = proc.stdout.decode("utf-8", "replace")
    lines = out.splitlines()
    try:
        end = max(i for i, line in enumerate(lines) if line.strip() == "___LOG_END___")
        start = max(
            i for i, line in enumerate(lines) if i < end and line.strip() == "___LOG_START___"
        )
    except ValueError as exc:
        tail = (out or proc.stderr.decode("utf-8", "replace"))[-400:]
        raise RuntimeError(f"ssh did not return the log ({proc.returncode}): {tail}") from exc
    return "\n".join(lines[start + 1 : end])


def fetch_local(cfg: dict) -> str:
    path = Path(cfg["log"])
    if not path.is_file():
        return ""
    return path.read_text(errors="replace")


class Board:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.lock = threading.Lock()
        self.snapshot = {
            "connected": False,
            "error": None,
            "fetched_at": None,
            **parse_log(""),
        }

    def loop(self) -> None:
        while True:
            started = time.time()
            try:
                text = fetch_remote(self.cfg) if self.cfg["ssh"] else fetch_local(self.cfg)
                parsed = parse_log(text)
                with self.lock:
                    self.snapshot = {
                        "connected": True,
                        "error": None,
                        "fetched_at": time.time(),
                        **parsed,
                    }
            except Exception as exc:
                with self.lock:
                    self.snapshot = {
                        **self.snapshot,
                        "connected": False,
                        "error": str(exc).splitlines()[-1][:300],
                        "fetched_at": time.time(),
                    }
            time.sleep(max(0.2, self.cfg["interval"] - (time.time() - started)))

    def get(self) -> dict:
        with self.lock:
            return dict(self.snapshot)


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Camões LSTM</title>
<style>
  :root { color-scheme: light; }
  body {
    margin: 0 auto;
    max-width: 40rem;
    padding: 1.4rem 1.1rem 3rem;
    background: #f4efe6;
    color: #1d1914;
    font: 16px/1.45 Georgia, "Iowan Old Style", Palatino, serif;
  }
  header { display: flex; justify-content: space-between; gap: 1rem; align-items: baseline; }
  h1 { font-size: 1.55rem; font-weight: 500; margin: 0; }
  .phase { font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.82rem; letter-spacing: 0.02em; }
  .phase.run { color: #9a3412; }
  .phase.done { color: #3f6212; }
  .phase.bad { color: #991b1b; }
  .bar { height: 6px; background: #e6dccb; margin: 0.9rem 0 1rem; }
  .bar > div { height: 100%; background: #9a3412; width: 0; }
  .nums {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
    font-family: ui-sans-serif, system-ui, sans-serif;
  }
  .num { background: #fffdf8; border: 1px solid #e4d8c4; padding: 0.65rem 0.75rem; }
  .num span { display: block; color: #6b6256; font-size: 0.75rem; }
  .num b { display: block; margin-top: 0.15rem; font-size: 1.25rem; font-variant-numeric: tabular-nums; font-weight: 600; }
  figure { margin: 1rem 0; }
  svg { width: 100%; height: 190px; background: #fffdf8; border: 1px solid #e4d8c4; display: block; }
  .legend { font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.78rem; color: #6b6256; margin-top: 0.35rem; }
  .swatch { display: inline-block; width: 0.7rem; height: 0.7rem; margin: 0 0.2rem 0 0.7rem; vertical-align: -1px; }
  h2 { font-size: 1rem; font-weight: 500; margin: 1.3rem 0 0.4rem; }
  pre {
    margin: 0;
    white-space: pre-wrap;
    font: inherit;
    background: #fffdf8;
    border: 1px solid #e4d8c4;
    padding: 0.9rem 1rem;
    min-height: 6rem;
  }
  footer { margin-top: 1rem; color: #6b6256; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.78rem; }
</style>
</head>
<body>
<header>
  <h1>Os Lusíadas</h1>
  <div id="phase" class="phase">Connecting…</div>
</header>
<div class="bar"><div id="fill"></div></div>
<section class="nums">
  <div class="num"><span>Step loss</span><b id="loss">–</b></div>
  <div class="num"><span>Train</span><b id="train">–</b></div>
  <div class="num"><span>Validation</span><b id="val">–</b></div>
</section>
<figure>
  <svg id="chart" viewBox="0 0 640 190" preserveAspectRatio="none"></svg>
  <div class="legend"><span class="swatch" style="background:#9a3412"></span>train <span class="swatch" style="background:#3f6212"></span>validation <span id="best"></span></div>
</figure>
<h2 id="sample-title">Latest sample</h2>
<pre id="sample"></pre>
<footer id="foot"></footer>
<script>
const fmt = (n, d=3) => (n === null || n === undefined) ? "–" : Number(n).toFixed(d);

function draw(epochs) {
  const svg = document.getElementById("chart");
  if (!epochs.length) { svg.innerHTML = ""; return; }
  const w = 640, h = 190, pad = 16;
  const ys = epochs.flatMap(e => [e.train, e.val]);
  let lo = Math.min(...ys), hi = Math.max(...ys);
  if (hi - lo < 0.05) hi = lo + 0.05;
  const x = i => epochs.length === 1 ? w / 2 : pad + i * (w - 2 * pad) / (epochs.length - 1);
  const y = v => pad + (hi - v) / (hi - lo) * (h - 2 * pad);
  const pts = key => epochs.map((e, i) => x(i).toFixed(1) + "," + y(e[key]).toFixed(1)).join(" ");
  svg.innerHTML =
    '<polyline fill="none" stroke="#9a3412" stroke-width="2.5" points="' + pts("train") + '"/>' +
    '<polyline fill="none" stroke="#3f6212" stroke-width="2.5" points="' + pts("val") + '"/>';
}

async function tick() {
  let data;
  try {
    data = await (await fetch("/api/status", {cache: "no-store"})).json();
  } catch (err) {
    document.getElementById("phase").textContent = "Page offline";
    document.getElementById("phase").className = "phase bad";
    return;
  }
  const phase = document.getElementById("phase");
  if (!data.connected && data.error) {
    phase.textContent = "No link to the trainer";
    phase.className = "phase bad";
  } else if (data.phase === "running") {
    phase.textContent = "Training · " + (data.step || 0) + " / " + (data.total_steps || "?");
    phase.className = "phase run";
  } else if (data.phase === "done") {
    phase.textContent = "Finished" + (data.elapsed_min ? " · " + fmt(data.elapsed_min, 1) + " min" : "");
    phase.className = "phase done";
  } else if (data.phase === "starting") {
    phase.textContent = "Starting";
    phase.className = "phase run";
  } else {
    phase.textContent = "Waiting for a log";
    phase.className = "phase";
  }
  const frac = data.total_steps ? Math.min(1, (data.step || 0) / data.total_steps) : (data.phase === "done" ? 1 : 0);
  document.getElementById("fill").style.width = (frac * 100) + "%";
  document.getElementById("loss").textContent = fmt(data.loss);
  document.getElementById("train").textContent = fmt(data.train);
  document.getElementById("val").textContent = fmt(data.val);
  document.getElementById("best").textContent = data.best_epoch ? " · lowest val " + fmt(data.best_val) + " at epoch " + data.best_epoch : "";
  document.getElementById("sample-title").textContent = data.sample_epoch ? "Sample at epoch " + data.sample_epoch : "Latest sample";
  document.getElementById("sample").textContent = data.sample || "No sample yet.";
  draw(data.epochs || []);
  const when = data.fetched_at ? new Date(data.fetched_at * 1000).toLocaleTimeString() : "–";
  const rate = data.steps_per_s ? " · " + fmt(data.steps_per_s, 1) + " steps/s" : "";
  const err = data.error ? " · " + data.error : "";
  document.getElementById("foot").textContent = "Updated " + when + rate + err;
}

tick();
setInterval(tick, 2000);
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    board: Board

    def log_message(self, fmt: str, *args) -> None:
        return

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = self.path.split("?", 1)[0]
        if path == "/api/status":
            body = json.dumps(self.board.get()).encode()
            self._send(200, body, "application/json; charset=utf-8")
            return
        if path in ("/", "/index.html"):
            self._send(200, PAGE.encode(), "text/html; charset=utf-8")
            return
        self._send(404, b"not found", "text/plain; charset=utf-8")


def main() -> None:
    cfg = config()
    if not cfg["ssh"] and not Path(cfg["log"]).is_file() and not os.environ.get("CAMOES_LOG"):
        missing = ROOT / "follow.env"
        if not missing.is_file():
            raise SystemExit(f"Create {missing} from follow.env.example")
    board = Board(cfg)
    threading.Thread(target=board.loop, name="log-poll", daemon=True).start()
    Handler.board = board
    server = ThreadingHTTPServer((cfg["bind"], cfg["port"]), Handler)
    where = "ssh " + cfg["ssh"] if cfg["ssh"] else cfg["log"]
    print(f"following {where}", flush=True)
    print(f"http://{cfg['bind']}:{cfg['port']}/", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
