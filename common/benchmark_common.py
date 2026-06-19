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


DATA_DIR_TOKEN = "{data_dir}"


def resolve_storage_paths(config: dict[str, Any], data_dir: str) -> list[str]:
    """Substitute the {data_dir} token in every storage path in a config in place.

    Covers the OffloadingConnector shared_storage_path, tiering
    secondary_tiers[].root_dir, and the LMCache LMCACHE_LOCAL_DISK env path.
    Returns the list of resolved paths (empty if none contained the token).
    """
    def sub(value: str) -> Optional[str]:
        return value.replace(DATA_DIR_TOKEN, data_dir) if DATA_DIR_TOKEN in value else None

    vllm_args = config.get("vllm_args", {})
    resolved: list[str] = []

    kv_transfer_cfg = vllm_args.get("--kv-transfer-config")
    if isinstance(kv_transfer_cfg, str):
        try:
            kv_transfer_cfg = json.loads(kv_transfer_cfg)
        except json.JSONDecodeError:
            kv_transfer_cfg = None

    if isinstance(kv_transfer_cfg, dict):
        extra = kv_transfer_cfg.get("kv_connector_extra_config", {})

        existing = extra.get("shared_storage_path") or kv_transfer_cfg.get("shared_storage_path", "")
        new_path = sub(existing) if existing else None
        if new_path:
            if "shared_storage_path" in extra:
                extra["shared_storage_path"] = new_path
                kv_transfer_cfg["kv_connector_extra_config"] = extra
            else:
                kv_transfer_cfg["shared_storage_path"] = new_path
            resolved.append(new_path)

        for tier in extra.get("secondary_tiers", []):
            if isinstance(tier, dict) and tier.get("root_dir"):
                new_root = sub(tier["root_dir"])
                if new_root:
                    tier["root_dir"] = new_root
                    resolved.append(new_root)

        vllm_args["--kv-transfer-config"] = kv_transfer_cfg

    env = config.get("env", {})
    disk = env.get("LMCACHE_LOCAL_DISK")
    if disk:
        new_disk = sub(disk)
        if new_disk:
            env["LMCACHE_LOCAL_DISK"] = new_disk
            resolved.append(new_disk)

    return resolved
