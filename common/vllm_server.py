"""Shared vLLM server lifecycle + benchmark subprocess helpers.

Both bench.py and scripts/pareto_measure.py launch a vLLM server, wait for it to
become healthy, run a benchmark subprocess against it, then shut it down. This
module is the single source of truth for that start/run/stop logic so the
robustness fixes (process-group signalling, two-SIGINT graceful shutdown, SIGKILL
fallback) live in exactly one place.
"""

import json
import os
import signal
import subprocess
import sys
import time
from typing import Any, Optional

import requests


def stringify_for_cli(value: Any) -> str:
    """Render a config value for the command line (JSON for dict/list)."""
    if isinstance(value, (dict, list)):
        return json.dumps(value)
    return str(value)


def build_vllm_command(
    *,
    model: str,
    max_model_len: int,
    gpu_mem_util: float,
    vllm_args: dict,
    port: int,
    launcher: Optional[list[str]] = None,
    model_as_flag: bool = True,
    extra_base_args: Optional[list[str]] = None,
    skip_keys: tuple = (),
) -> list[str]:
    """Build a vLLM launch command.

    launcher        : argv prefix (default: `python -m vllm.entrypoints...`).
    model_as_flag   : True -> `--model M`; False -> positional `M` (vllm serve M).
    extra_base_args  : appended after the standard base flags (e.g. kv-cache-dtype).
    skip_keys       : vllm_args keys to ignore (e.g. internal "is_storage").
    """
    if launcher is None:
        launcher = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    cmd = list(launcher)
    if model_as_flag:
        cmd += ["--model", model]
    else:
        cmd += [model]
    cmd += [
        "--port", str(port),
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_mem_util),
    ]
    if extra_base_args:
        cmd += list(extra_base_args)
    for flag, value in vllm_args.items():
        if flag in skip_keys:
            continue
        cmd.append(flag)
        if value is not None:
            cmd.append(stringify_for_cli(value))
    return cmd


def wait_for_server(port: int, timeout: int) -> bool:
    """Poll /health until the server is ready or timeout elapses."""
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=3).status_code == 200:
                return True
        except requests.exceptions.RequestException:
            pass
        time.sleep(2)
    return False


def start_server(cmd: list[str], log_path: str, env: Optional[dict] = None) -> subprocess.Popen:
    """Launch a prebuilt vLLM command, teeing stdout/stderr to log_path.

    start_new_session=True puts the server (and every child it spawns: EngineCore,
    the LMCache worker that pins the CUDA kv_buffer, etc.) into its own process
    group/session so stop_server() can signal the whole tree via killpg. Without
    it, a SIGKILL on the parent orphans the GPU-holding children and leaks VRAM
    into the next run.
    """
    # Append mode so both tee streams land in the same file.
    tee_out = subprocess.Popen(["tee", "-a", log_path], stdin=subprocess.PIPE)
    tee_err = subprocess.Popen(["tee", "-a", log_path], stdin=subprocess.PIPE)
    return subprocess.Popen(
        cmd, env=env, stdout=tee_out.stdin, stderr=tee_err.stdin,
        start_new_session=True,
    )


def stop_server(proc: subprocess.Popen) -> None:
    """Gracefully stop a vLLM server: two SIGINTs (10s grace each), then SIGKILL.

    vLLM often needs two SIGINTs to fully shut down its engine subprocess (mimics
    two Ctrl+C). Signals go to the whole process group (see start_new_session in
    start_server) so children holding GPU memory die too; signalling only the
    parent PID orphans them and leaks VRAM into the next run.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        pgid = None

    def _signal_group(sig: int) -> None:
        if pgid is not None:
            try:
                os.killpg(pgid, sig)
                return
            except ProcessLookupError:
                return
        proc.send_signal(sig)

    if proc.poll() is None:
        for attempt in range(2):
            _signal_group(signal.SIGINT)
            try:
                proc.wait(timeout=10)
                break
            except subprocess.TimeoutExpired:
                print(f"  …  vLLM still running after SIGINT #{attempt + 1}, waiting…")
        else:
            print("  ⚠  vLLM did not exit after 2 SIGINTs, sending SIGKILL.")
            _signal_group(signal.SIGKILL)
            proc.wait()
    # Parent is gone, but stragglers in the group (e.g. an LMCache worker that
    # ignored SIGINT) may still hold VRAM. Reap the whole group regardless.
    _signal_group(signal.SIGKILL)
    print("  ■  vLLM server stopped.")


def run_benchmark_subprocess(cmd: list[str]) -> subprocess.CompletedProcess:
    """Run a benchmark command to completion, echoing its output."""
    print(f"\n  ▶  Running benchmark: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout + ("\n" + result.stderr if result.stderr.strip() else ""))
    if result.returncode != 0:
        print(f"  ✗  Benchmark exited with code {result.returncode}")
    return result
