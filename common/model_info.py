"""Fetch model KV-cache geometry from a Hugging Face config.json.

Mapping to config.json fields:
  num_layers   -> num_hidden_layers
  num_kv_heads -> num_key_value_heads  (falls back to num_attention_heads, i.e. MHA)
  head_dim     -> explicit head_dim if present, else hidden_size // num_attention_heads

Multimodal checkpoints nest the language-model params under "text_config"; we
prefer that block when present.
"""

import json
from pathlib import Path

# Reuse the locked, config-only, offline-first resolver already used by the
# kv-connector harness so we don't duplicate the download/lock logic.
from common.kv_connector_harness import _resolve_local_hf_config_path_cached


def _text_config(cfg: dict) -> dict:
    sub = cfg.get("text_config")
    return sub if isinstance(sub, dict) else cfg


def fetch_model_geometry(model_id: str) -> tuple[int, int, int]:
    """Return (num_layers, num_kv_heads, head_dim) for model_id.

    Resolves config.json from a local path, the local HF cache, or a remote
    config-only download (in that order), then extracts the geometry.
    """
    config_dir = _resolve_local_hf_config_path_cached(model_id, None)
    with open(Path(config_dir) / "config.json") as f:
        cfg = _text_config(json.load(f))

    num_layers = int(cfg["num_hidden_layers"])
    num_attention_heads = int(cfg["num_attention_heads"])
    num_kv_heads = int(cfg.get("num_key_value_heads", num_attention_heads))
    head_dim = int(cfg.get("head_dim") or cfg["hidden_size"] // num_attention_heads)

    return num_layers, num_kv_heads, head_dim
