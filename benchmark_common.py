import csv
import json
import os
import shutil
from dataclasses import asdict
from typing import Any, Optional


GPU_TRANSFER_FIELDS = ["direction", "ts_us", "dur_us", "num_bytes"]


def pct(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    idx = p / 100.0 * (len(s) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(s) - 1)
    return s[lo] + (idx - lo) * (s[hi] - s[lo])


def parse_profile_json(json_path: str) -> list[dict]:
    with open(json_path) as f:
        data = json.load(f)
    events = data.get("traceEvents", data) if isinstance(data, dict) else data

    transfers = []
    for event in events:
        if not isinstance(event, dict):
            continue
        name = event.get("name", "")
        if name.startswith("cuda_transfer("):
            inner = name.split("(", 1)[1].rstrip(")")
            direction = "to_gpu" if "cpu_to_gpu" in inner else "from_gpu"
        elif name == "VLLMPagedMemGPUConnectorV2.to_gpu.kernel":
            direction = "to_gpu"
        elif name == "VLLMPagedMemGPUConnectorV2.from_gpu.kernel":
            direction = "from_gpu"
        else:
            continue
        transfers.append({
            "direction": direction,
            "ts_us": event.get("ts", ""),
            "dur_us": event.get("dur", ""),
            "num_bytes": (event.get("args") or {}).get("num_bytes", ""),
        })
    return transfers


def write_gpu_transfer_csv(transfers: list[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GPU_TRANSFER_FIELDS)
        writer.writeheader()
        writer.writerows(transfers)


def save_profile_artifacts(
    profile_json: Optional[str],
    output_dir: str,
    output_stem: str,
) -> Optional[str]:
    if not profile_json or not os.path.exists(profile_json):
        return None
    saved_json = os.path.join(output_dir, f"{output_stem}_profile.json")
    shutil.copy2(profile_json, saved_json)
    transfers = parse_profile_json(saved_json)
    if not transfers:
        return None
    transfer_csv = os.path.join(output_dir, f"{output_stem}_gpu_transfers.csv")
    write_gpu_transfer_csv(transfers, transfer_csv)
    return transfer_csv


def write_dataclass_csv(rows: list[Any], path: str) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].__dataclass_fields__.keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def resolve_scratch_path_for_config(config: dict[str, Any], override_path: Optional[str]) -> Optional[str]:
    vllm_args = config.get("vllm_args", {})
    kv_transfer_cfg = vllm_args.get("--kv-transfer-config")
    if kv_transfer_cfg is None:
        return None
    if isinstance(kv_transfer_cfg, str):
        try:
            kv_transfer_cfg = json.loads(kv_transfer_cfg)
        except json.JSONDecodeError:
            return None

    extra = kv_transfer_cfg.get("kv_connector_extra_config", {})
    existing = extra.get("shared_storage_path", "")
    if override_path:
        resolved = override_path
    elif existing.startswith("/scratch-node/"):
        scratch_root = "/scratch-node"
        try:
            entries = os.listdir(scratch_root)
        except OSError:
            print(f"  WARNING: could not list {scratch_root}")
            return None
        matching = sorted(e for e in entries if e.startswith("jkanichai"))
        if not matching:
            print(f"  WARNING: no jkanichai* directory found in {scratch_root}")
            return None
        suffix = existing[len("/scratch-node/"):].split("/", 1)
        if len(suffix) == 1:
            return None
        resolved = os.path.join(scratch_root, matching[0], suffix[1].lstrip("/"))
    else:
        return None

    extra["shared_storage_path"] = resolved
    kv_transfer_cfg["kv_connector_extra_config"] = extra
    vllm_args["--kv-transfer-config"] = kv_transfer_cfg
    return resolved
