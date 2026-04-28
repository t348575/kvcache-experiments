#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import random
import time
import fcntl
import hashlib
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

MODEL_DTYPE = "float16"
CACHE_DTYPE = "auto"
DEVICE = "cuda"
MAX_NUM_BATCHED_TOKENS = 131072
MAX_NUM_SEQS = 16
MAX_MODEL_LEN = 131072
REGISTER_CACHE_MODE = "auto"
CROSS_LAYER_BACKEND_PATH = "vllm.v1.attention.backends.flash_attn:FlashAttentionBackend"


@dataclass(frozen=True)
class RuntimeConnectorConfig:
    connector_name: str
    connector_module_path: str | None = None
    connector_extra_config: dict[str, Any] | None = None
    model_name: str = "meta-llama/Llama-3.2-3B-Instruct"
    hf_config_path: str | None = None
    block_size: int = 16
    num_blocks: int = 4096
    max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS
    max_num_seqs: int = MAX_NUM_SEQS
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = False
    enable_permute_local_kv: bool = False
    kv_role: str = "kv_both"
    async_scheduling: bool = False
    kv_load_failure_policy: str = "fail"


@dataclass(frozen=True)
class ScenarioRequestSpec:
    request_id: int
    doc_tokens: int
    reuse_source_id: int | None
    reuse_prefix_len: int
    prompt_token_ids: tuple[int, ...]
    logical_request_id: int | None = None
    prompt_key: str | None = None
    traffic_scope: str = "default"
    target_instance: int | None = None


@dataclass
class ScenarioRequestResult:
    request_id: int
    doc_tokens: int
    reuse_source_id: int | None
    reuse_prefix_len: int
    logical_request_id: int | None
    prompt_key: str | None
    traffic_scope: str
    target_instance: int | None
    scheduler_steps: int
    total_scheduled_tokens: int
    max_scheduled_tokens_per_step: int
    connector_load_ops: int
    connector_load_bytes: int
    connector_load_time_s: float
    connector_store_ops: int
    connector_store_bytes: int
    connector_store_time_s: float
    finished_sending: bool
    finished_recving: bool
    successful: bool
    duration_s: float
    error: str | None = None


@dataclass(frozen=True)
class RuntimePlacement:
    runtime_kv_device: str
    runtime_kv_medium: str
    allocated_on_cpu: bool


@contextmanager
def _temporary_env(updates: dict[str, str]):
    previous = {key: os.environ.get(key) for key in updates}
    try:
        os.environ.update(updates)
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _normalize_hf_config_path(path: str) -> str:
    candidate = Path(path).expanduser().resolve()
    if not candidate.exists():
        raise FileNotFoundError(f"HF config path does not exist: {candidate}")
    if candidate.is_file():
        if candidate.name != "config.json":
            raise ValueError(
                "HF config path must be a directory or a config.json file: "
                f"{candidate}"
            )
        return str(candidate.parent)
    config_file = candidate / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"No config.json found under HF config path: {candidate}")
    return str(candidate)


def _download_hf_config_once(model_name: str) -> str:
    try:
        from huggingface_hub import snapshot_download
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Resolving a remote model config requires huggingface_hub in the active environment"
        ) from exc

    lock_dir = Path.home() / ".cache" / "kvcache-experiments" / "hf-config-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(model_name.encode("utf-8")).hexdigest() + ".lock"
    lock_path = lock_dir / lock_name

    with open(lock_path, "a+") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        try:
            snapshot_path = snapshot_download(
                repo_id=model_name,
                allow_patterns=["config.json"],
                local_files_only=True,
            )
        except Exception:
            snapshot_path = snapshot_download(
                repo_id=model_name,
                allow_patterns=["config.json"],
            )
    return _normalize_hf_config_path(snapshot_path)


@lru_cache(maxsize=None)
def _resolve_local_hf_config_path_cached(
    model_name: str, hf_config_path: str | None
) -> str:
    if hf_config_path:
        return _normalize_hf_config_path(hf_config_path)

    model_path = Path(model_name).expanduser()
    if model_path.exists():
        return _normalize_hf_config_path(str(model_path))

    return _download_hf_config_once(model_name)


def resolve_local_hf_config_path(config: RuntimeConnectorConfig) -> str:
    return _resolve_local_hf_config_path_cached(config.model_name, config.hf_config_path)


def build_model_config_kwargs(config: RuntimeConnectorConfig) -> dict[str, Any]:
    local_hf_config_path = resolve_local_hf_config_path(config)
    return {
        "model": local_hf_config_path,
        "hf_config_path": local_hf_config_path,
        "trust_remote_code": True,
        "dtype": MODEL_DTYPE,
        "seed": 42,
        "hf_overrides": {},
        "config_format": "hf",
    }


def make_model_config(config: RuntimeConnectorConfig) -> Any:
    from vllm.config import ModelConfig

    with _temporary_env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}):
        return ModelConfig(**build_model_config_kwargs(config))


def _resolve_runtime_placement(extra_config: dict[str, Any] | None) -> RuntimePlacement:
    connector_extra_config = extra_config or {}
    skip_gpu_copy = bool(connector_extra_config.get("skip_gpu_copy", False))
    runtime_kv_device = str(
        connector_extra_config.get(
            "runtime_kv_device",
            "cpu" if skip_gpu_copy else DEVICE,
        )
    )
    runtime_kv_medium = str(
        connector_extra_config.get(
            "runtime_kv_medium",
            "cpu" if runtime_kv_device == "cpu" else "gpu",
        )
    ).strip().lower()
    allocated_on_cpu = runtime_kv_device == "cpu"
    return RuntimePlacement(
        runtime_kv_device=runtime_kv_device,
        runtime_kv_medium=runtime_kv_medium,
        allocated_on_cpu=allocated_on_cpu,
    )

def _should_use_metadata_only_dummy_kv(extra_config: dict[str, Any] | None) -> bool:
    connector_extra_config = extra_config or {}
    return bool(connector_extra_config.get("skip_gpu_copy", False)) and str(
        connector_extra_config.get("file_io_mode", "full")
    ).strip().lower() == "metadata_only"


def _allocate_minimal_uniform_kv_caches(
    *,
    torch: Any,
    kv_cache_spec: Any,
    cache_dtype: str,
    device: Any,
    kernel_block_size: int,
    layer_names: list[str],
    attn_backend: Any,
) -> tuple[dict[str, Any], Any, Any]:
    num_layers = len(layer_names)
    num_blocks = 1
    num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size
    kernel_num_blocks = num_blocks * num_blocks_per_kv_block
    kv_cache_shape = attn_backend.get_kv_cache_shape(
        kernel_num_blocks,
        kernel_block_size,
        kv_cache_spec.num_kv_heads,
        kv_cache_spec.head_size,
        cache_dtype_str=cache_dtype,
    )
    kv_cache_shape = (num_layers,) + kv_cache_shape

    try:
        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order(
            include_num_layers_dimension=True
        )
        assert len(kv_cache_stride_order) == len(kv_cache_shape)
    except (AttributeError, NotImplementedError):
        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))

    kv_cache_shape = tuple(kv_cache_shape[i] for i in kv_cache_stride_order)
    total_size = kv_cache_spec.page_size_bytes * num_layers
    cross_layers_kv_cache = (
        torch.zeros(total_size, dtype=torch.int8, device=device)
        .view(kv_cache_spec.dtype)
        .view(kv_cache_shape)
    )
    inv_order = [
        kv_cache_stride_order.index(i) for i in range(len(kv_cache_stride_order))
    ]
    permuted_kv_cache = cross_layers_kv_cache.permute(*inv_order)
    kv_caches = {
        layer_name: permuted_kv_cache[layer_idx]
        for layer_idx, layer_name in enumerate(layer_names)
    }
    return kv_caches, cross_layers_kv_cache, attn_backend


def estimate_kv_cache_bytes(config: RuntimeConnectorConfig) -> int:
    placement = _resolve_runtime_placement(config.connector_extra_config)
    try:
        import torch
        from vllm.config import (
            AttentionConfig,
            CacheConfig,
            DeviceConfig,
            KVTransferConfig,
            SchedulerConfig,
            VllmConfig,
        )
        from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheTensor
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "estimate_kv_cache_bytes requires both 'torch' and 'vllm' in the active environment"
        ) from exc

    dtype = getattr(torch, MODEL_DTYPE, None)
    if dtype is None:
        raise ValueError(f"unsupported torch dtype: {MODEL_DTYPE}")

    model_config = make_model_config(config)
    scheduler_config = SchedulerConfig(
        max_num_seqs=config.max_num_seqs,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_model_len=MAX_MODEL_LEN,
        enable_chunked_prefill=config.enable_chunked_prefill,
        is_encoder_decoder=model_config.is_encoder_decoder,
        async_scheduling=config.async_scheduling,
        disable_hybrid_kv_cache_manager=True,
    )
    cache_config = CacheConfig(
        block_size=config.block_size,
        gpu_memory_utilization=0.9,
        cache_dtype=CACHE_DTYPE,
        enable_prefix_caching=config.enable_prefix_caching,
    )
    kv_transfer_config = KVTransferConfig(
        kv_connector=config.connector_name,
        kv_connector_module_path=config.connector_module_path,
        kv_role=config.kv_role,
        enable_permute_local_kv=config.enable_permute_local_kv,
        kv_connector_extra_config={
            **(config.connector_extra_config or {}),
            "runtime_kv_medium": placement.runtime_kv_medium,
        },
        kv_load_failure_policy=config.kv_load_failure_policy,
    )
    vllm_config = VllmConfig(
        scheduler_config=scheduler_config,
        model_config=model_config,
        cache_config=cache_config,
        kv_transfer_config=kv_transfer_config,
        device_config=DeviceConfig(placement.runtime_kv_device),
        attention_config=AttentionConfig(),
    )
    parallel_config = vllm_config.parallel_config
    num_layers = model_config.get_num_layers(parallel_config)
    num_kv_heads = model_config.get_num_kv_heads(parallel_config)
    head_size = model_config.get_head_size()
    kv_cache_spec = FullAttentionSpec(
        block_size=config.block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )
    kv_cache_tensor = KVCacheTensor(
        size=kv_cache_spec.page_size_bytes * config.num_blocks,
        shared_by=["layer0"],
    )
    kv_cache_config = KVCacheConfig(
        num_blocks=config.num_blocks,
        kv_cache_tensors=[kv_cache_tensor for _ in range(num_layers)],
        kv_cache_groups=[],
    )
    return sum(tensor.size for tensor in kv_cache_config.kv_cache_tensors)


def get_free_cuda_bytes() -> list[int]:
    try:
        import torch
    except ModuleNotFoundError as exc:
        raise RuntimeError("get_free_cuda_bytes requires torch in the active environment") from exc

    if not torch.cuda.is_available():
        return []
    free_bytes: list[int] = []
    for device_idx in range(torch.cuda.device_count()):
        with torch.cuda.device(device_idx):
            free_bytes.append(int(torch.cuda.mem_get_info()[0]))
    return free_bytes


def parse_int_range(value: str) -> tuple[int, int]:
    if "-" in value:
        lo_str, hi_str = value.split("-", 1)
        lo, hi = int(lo_str), int(hi_str)
        return (lo, hi) if lo <= hi else (hi, lo)
    number = int(value)
    return number, number


def parse_float_range(value: str) -> tuple[float, float]:
    if "-" in value:
        lo_str, hi_str = value.split("-", 1)
        lo, hi = float(lo_str), float(hi_str)
        return (lo, hi) if lo <= hi else (hi, lo)
    number = float(value)
    return number, number


def parse_json_dict(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    parsed = json.loads(value)
    if not isinstance(parsed, dict):
        raise ValueError("connector extra config must be a JSON object")
    return parsed


def build_request_specs(
    *,
    num_requests: int,
    doc_size_range: tuple[int, int],
    store_pct: float,
    prefix_reuse_range: tuple[float, float],
    seed: int,
    request_id_base: int = 0,
) -> list[ScenarioRequestSpec]:
    if not 0.0 <= store_pct <= 1.0:
        raise ValueError("store_pct must be in [0.0, 1.0]")

    rng = random.Random(seed)
    requests: list[ScenarioRequestSpec] = []

    for local_request_id in range(num_requests):
        request_id = request_id_base + local_request_id
        doc_tokens = rng.randint(*doc_size_range)
        corpus_doc = ScenarioRequestSpec(
            request_id=request_id,
            doc_tokens=doc_tokens,
            reuse_source_id=None,
            reuse_prefix_len=0,
            prompt_token_ids=tuple(
                _corpus_token(request_id, token_idx) for token_idx in range(doc_tokens)
            ),
        )
        should_store = rng.random() < store_pct
        if should_store:
            request = ScenarioRequestSpec(
                request_id=request_id,
                doc_tokens=doc_tokens,
                reuse_source_id=None,
                reuse_prefix_len=0,
                prompt_token_ids=corpus_doc.prompt_token_ids,
                logical_request_id=request_id,
                prompt_key=f"request-{request_id}",
            )
            requests.append(request)
            continue

        source = corpus_doc
        reuse_frac = rng.uniform(*prefix_reuse_range)
        reuse_prefix_len = min(
            doc_tokens,
            source.doc_tokens,
            max(1, int(source.doc_tokens * reuse_frac)),
        )
        suffix_len = max(0, doc_tokens - reuse_prefix_len)
        prompt_token_ids = source.prompt_token_ids[:reuse_prefix_len] + tuple(
            _unique_token(request_id, token_idx + reuse_prefix_len)
            for token_idx in range(suffix_len)
        )
        requests.append(
            ScenarioRequestSpec(
                request_id=request_id,
                doc_tokens=doc_tokens,
                reuse_source_id=source.request_id,
                reuse_prefix_len=reuse_prefix_len,
                prompt_token_ids=prompt_token_ids,
                logical_request_id=request_id,
                prompt_key=f"request-{request_id}",
            )
        )

    return requests


def results_to_rows(results: list[ScenarioRequestResult]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def _unique_token(request_id: int, token_idx: int) -> int:
    return request_id * 1_000_000 + token_idx + 1


def _corpus_token(corpus_doc_id: int, token_idx: int) -> int:
    return corpus_doc_id * 1_000_000 + token_idx + 1


class RuntimeKVConnectorHarness:
    def __init__(self, config: RuntimeConnectorConfig):
        self.config = config
        self._runtime: dict[str, Any] | None = None
        self._placement = _resolve_runtime_placement(config.connector_extra_config)

    @property
    def placement(self) -> RuntimePlacement:
        return self._placement

    def _ensure_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime

        try:
            import importlib

            import torch
            from vllm import SamplingParams
            from vllm.config import (
                AttentionConfig,
                CacheConfig,
                DeviceConfig,
                KVTransferConfig,
                SchedulerConfig,
                VllmConfig,
                set_current_vllm_config,
            )
            from vllm.distributed.kv_transfer.kv_connector.factory import (
                KVConnectorFactory,
            )
            from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
            from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
                OffloadingConnectorMetadata,
            )
            from vllm.forward_context import ForwardContext
            from vllm.v1.core.sched.output import CachedRequestData
            from vllm.v1.core.sched.scheduler import Scheduler
            from vllm.v1.kv_cache_interface import (
                FullAttentionSpec,
                KVCacheConfig,
                KVCacheGroupSpec,
                KVCacheTensor,
            )
            from vllm.v1.outputs import (
                EMPTY_MODEL_RUNNER_OUTPUT,
                KVConnectorOutput,
                ModelRunnerOutput,
            )
            from vllm.v1.request import Request
            from vllm.v1.structured_output import StructuredOutputManager
            from vllm.v1.worker.kv_connector_model_runner_mixin import (
                KVConnectorModelRunnerMixin,
            )
            from vllm.v1.worker.utils import AttentionGroup
            from vllm.v1.core.kv_cache_utils import (
                get_request_block_hasher,
                init_none_hash,
            )
            from vllm.utils.hashing import sha256
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "runtime connector harness requires both 'torch' and 'vllm' in the active environment"
            ) from exc

        dtype = getattr(torch, MODEL_DTYPE, None)
        if dtype is None:
            raise ValueError(f"unsupported torch dtype: {MODEL_DTYPE}")

        device = torch.device(self._placement.runtime_kv_device)
        connector_extra_config = dict(self.config.connector_extra_config or {})
        connector_extra_config["runtime_kv_device"] = self._placement.runtime_kv_device
        connector_extra_config["runtime_kv_medium"] = self._placement.runtime_kv_medium

        model_config = make_model_config(self.config)
        scheduler_config = SchedulerConfig(
            max_num_seqs=self.config.max_num_seqs,
            max_num_batched_tokens=self.config.max_num_batched_tokens,
            max_model_len=MAX_MODEL_LEN,
            enable_chunked_prefill=self.config.enable_chunked_prefill,
            is_encoder_decoder=model_config.is_encoder_decoder,
            async_scheduling=self.config.async_scheduling,
            disable_hybrid_kv_cache_manager=True,
        )
        cache_config = CacheConfig(
            block_size=self.config.block_size,
            gpu_memory_utilization=0.9,
            cache_dtype=CACHE_DTYPE,
            enable_prefix_caching=self.config.enable_prefix_caching,
        )
        kv_transfer_config = KVTransferConfig(
            kv_connector=self.config.connector_name,
            kv_connector_module_path=self.config.connector_module_path,
            kv_role=self.config.kv_role,
            enable_permute_local_kv=self.config.enable_permute_local_kv,
            kv_connector_extra_config=connector_extra_config,
            kv_load_failure_policy=self.config.kv_load_failure_policy,
        )
        vllm_config = VllmConfig(
            scheduler_config=scheduler_config,
            model_config=model_config,
            cache_config=cache_config,
            kv_transfer_config=kv_transfer_config,
            device_config=DeviceConfig(DEVICE),
            attention_config=AttentionConfig(),
        )

        parallel_config = vllm_config.parallel_config
        num_layers = model_config.get_num_layers(parallel_config)
        num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        head_size = model_config.get_head_size()
        layer_names = [f"layer{i}" for i in range(num_layers)]

        kv_cache_spec = FullAttentionSpec(
            block_size=self.config.block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            dtype=dtype,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=self.config.num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=kv_cache_spec.page_size_bytes * self.config.num_blocks,
                    shared_by=[layer_name],
                )
                for layer_name in layer_names
            ],
            kv_cache_groups=[KVCacheGroupSpec(layer_names, kv_cache_spec)],
        )
        vllm_config.cache_config.num_gpu_blocks = self.config.num_blocks

        scheduler = Scheduler(
            vllm_config=vllm_config,
            kv_cache_config=kv_cache_config,
            log_stats=True,
            structured_output_manager=StructuredOutputManager(vllm_config),
            block_size=self.config.block_size,
        )
        worker_connector = KVConnectorFactory.create_connector(
            vllm_config,
            KVConnectorRole.WORKER,
            kv_cache_config,
        )
        scheduler_connector = scheduler.connector
        if scheduler_connector is None:
            raise RuntimeError("scheduler did not create a KV connector")

        cache_mode = REGISTER_CACHE_MODE
        if cache_mode == "auto":
            cache_mode = (
                "cross_layer"
                if hasattr(worker_connector, "register_cross_layers_kv_cache")
                else "layer_dict"
            )

        module_name, _, class_name = CROSS_LAYER_BACKEND_PATH.partition(":")
        backend_module = importlib.import_module(module_name)
        backend_cls = getattr(backend_module, class_name)

        if cache_mode == "cross_layer":
            with set_current_vllm_config(vllm_config):
                if _should_use_metadata_only_dummy_kv(self.config.connector_extra_config):
                    _, cross_layers_kv_cache, _ = _allocate_minimal_uniform_kv_caches(
                        torch=torch,
                        kv_cache_spec=kv_cache_spec,
                        cache_dtype=CACHE_DTYPE,
                        device=device,
                        kernel_block_size=self.config.block_size,
                        layer_names=layer_names,
                        attn_backend=backend_cls,
                    )
                else:
                    _, cross_layers_kv_cache, _ = (
                        KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(
                            kv_cache_config=kv_cache_config,
                            attn_groups=[
                                [
                                    AttentionGroup(
                                        backend=backend_cls,
                                        layer_names=layer_names,
                                        kv_cache_spec=kv_cache_spec,
                                        kv_cache_group_id=0,
                                    )
                                ]
                            ],
                            cache_dtype=CACHE_DTYPE,
                            device=device,
                            kernel_block_sizes=[self.config.block_size],
                        )
                    )
                worker_connector.register_cross_layers_kv_cache(
                    kv_cache=cross_layers_kv_cache,
                    attn_backend=backend_cls,
                )
        elif cache_mode == "layer_dict":
            if _should_use_metadata_only_dummy_kv(self.config.connector_extra_config):
                kv_caches, _, _ = _allocate_minimal_uniform_kv_caches(
                    torch=torch,
                    kv_cache_spec=kv_cache_spec,
                    cache_dtype=CACHE_DTYPE,
                    device=device,
                    kernel_block_size=self.config.block_size,
                    layer_names=layer_names,
                    attn_backend=backend_cls,
                )
            else:
                kv_caches, _, _ = KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(
                    kv_cache_config=kv_cache_config,
                    attn_groups=[
                        [
                            AttentionGroup(
                                backend=backend_cls,
                                layer_names=layer_names,
                                kv_cache_spec=kv_cache_spec,
                                kv_cache_group_id=0,
                            )
                        ]
                    ],
                    cache_dtype=CACHE_DTYPE,
                    device=device,
                    kernel_block_sizes=[self.config.block_size],
                )
            worker_connector.register_kv_caches(kv_caches)
        else:
            raise ValueError(
                "register_cache_mode must be one of: auto, cross_layer, layer_dict"
            )

        os.environ.setdefault("PYTHONHASHSEED", "0")
        init_none_hash(sha256)
        dummy_ctx = ForwardContext(
            no_compile_layers={}, attn_metadata={}, slot_mapping={}
        )

        self._runtime = {
            "torch": torch,
            "SamplingParams": SamplingParams,
            "CachedRequestData": CachedRequestData,
            "EMPTY_MODEL_RUNNER_OUTPUT": EMPTY_MODEL_RUNNER_OUTPUT,
            "KVConnectorOutput": KVConnectorOutput,
            "ModelRunnerOutput": ModelRunnerOutput,
            "OffloadingConnectorMetadata": OffloadingConnectorMetadata,
            "Request": Request,
            "get_request_block_hasher": get_request_block_hasher,
            "sha256": sha256,
            "scheduler": scheduler,
            "scheduler_connector": scheduler_connector,
            "worker_connector": worker_connector,
            "dummy_ctx": dummy_ctx,
            "placement": self._placement,
        }
        return self._runtime

    def _make_request(self, spec: ScenarioRequestSpec):
        runtime = self._ensure_runtime()
        SamplingParams = runtime["SamplingParams"]
        Request = runtime["Request"]
        get_request_block_hasher = runtime["get_request_block_hasher"]

        sampling_params = SamplingParams(max_tokens=1, temperature=0.0)
        sampling_params.update_from_generation_config({}, 0)
        return Request(
            request_id=f"req-{spec.request_id}",
            prompt_token_ids=list(spec.prompt_token_ids),
            sampling_params=sampling_params,
            pooling_params=None,
            block_hasher=get_request_block_hasher(
                self.config.block_size, runtime["sha256"]
            ),
        )

    def _make_model_runner_output(
        self,
        reqs: list[Any],
        *,
        finished_sending: set[str] | None,
        finished_recving: set[str] | None,
    ):
        runtime = self._ensure_runtime()
        ModelRunnerOutput = runtime["ModelRunnerOutput"]
        KVConnectorOutput = runtime["KVConnectorOutput"]

        req_ids = [req.request_id for req in reqs]
        kv_connector_output = KVConnectorOutput(
            finished_sending=finished_sending,
            finished_recving=finished_recving,
            invalid_block_ids=set(),
        )
        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index={req_id: idx for idx, req_id in enumerate(req_ids)},
            sampled_token_ids=[[] if req.is_prefill_chunk else [0] for req in reqs],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=None,
            kv_connector_output=kv_connector_output,
        )

    def _get_pending_store_jobs(
        self, worker_connector: Any, req_key: str
    ) -> set[int] | None:
        connector_worker = getattr(worker_connector, "connector_worker", None)
        if connector_worker is None:
            return None
        store_jobs = getattr(connector_worker, "_store_jobs", None)
        if store_jobs is None:
            return None
        pending = store_jobs.get(req_key)
        return set(pending) if pending is not None else None

    def _empty_connector_stats_summary(self) -> dict[str, int | float]:
        return {
            "connector_load_ops": 0,
            "connector_load_bytes": 0,
            "connector_load_time_s": 0.0,
            "connector_store_ops": 0,
            "connector_store_bytes": 0,
            "connector_store_time_s": 0.0,
        }

    def _collect_connector_stats(self, worker_connector: Any) -> dict[str, int | float]:
        stats = worker_connector.get_kv_connector_stats()
        summary = self._empty_connector_stats_summary()
        if stats is None or stats.is_empty():
            return summary

        for transfer_type, ops in stats.data.items():
            if not isinstance(ops, list):
                continue
            total_bytes = 0
            total_time = 0.0
            for op in ops:
                if hasattr(op, "op_size"):
                    total_bytes += int(op.op_size)
                    total_time += float(op.op_time)
                elif isinstance(op, dict):
                    total_bytes += int(op.get("op_size", 0))
                    total_time += float(op.get("op_time", 0.0))

            if transfer_type.endswith("_to_GPU"):
                summary["connector_load_ops"] += len(ops)
                summary["connector_load_bytes"] += total_bytes
                summary["connector_load_time_s"] += total_time
            elif transfer_type.startswith("GPU_to_"):
                summary["connector_store_ops"] += len(ops)
                summary["connector_store_bytes"] += total_bytes
                summary["connector_store_time_s"] += total_time

        return summary

    def run_request(self, spec: ScenarioRequestSpec) -> ScenarioRequestResult:
        runtime = self._ensure_runtime()
        scheduler = runtime["scheduler"]
        worker_connector = runtime["worker_connector"]
        dummy_ctx = runtime["dummy_ctx"]
        EMPTY_MODEL_RUNNER_OUTPUT = runtime["EMPTY_MODEL_RUNNER_OUTPUT"]
        KVConnectorOutput = runtime["KVConnectorOutput"]
        OffloadingConnectorMetadata = runtime["OffloadingConnectorMetadata"]

        request = self._make_request(spec)
        req_key = request.request_id
        finished_sending_all: set[str] = set()
        finished_recving_all: set[str] = set()
        scheduler_steps = 0
        total_scheduled_tokens = 0
        max_scheduled_tokens_per_step = 0
        connector_stats_summary = self._empty_connector_stats_summary()
        start = time.time()

        try:
            scheduler.add_request(request)

            while scheduler.requests:
                scheduler_output = scheduler.schedule()
                if scheduler_output.total_num_scheduled_tokens > 0:
                    scheduler_steps += 1
                    total_scheduled_tokens += (
                        scheduler_output.total_num_scheduled_tokens
                    )
                    max_scheduled_tokens_per_step = max(
                        max_scheduled_tokens_per_step,
                        scheduler_output.total_num_scheduled_tokens,
                    )
                kv_connector_metadata = scheduler_output.kv_connector_metadata
                if kv_connector_metadata is not None:
                    worker_connector.handle_preemptions(kv_connector_metadata)
                    worker_connector.bind_connector_metadata(kv_connector_metadata)
                    worker_connector.start_load_kv(dummy_ctx)

                if scheduler_output.total_num_scheduled_tokens > 0:
                    worker_connector.wait_for_save()

                finished_sending, finished_recving = worker_connector.get_finished(
                    scheduler_output.finished_req_ids
                )
                finished_sending_all |= finished_sending
                finished_recving_all |= finished_recving
                worker_connector.clear_connector_metadata()

                model_runner_output = self._make_model_runner_output(
                    scheduler.running,
                    finished_sending=finished_sending,
                    finished_recving=finished_recving,
                )
                scheduler.update_from_output(scheduler_output, model_runner_output)

            flush_metadata = OffloadingConnectorMetadata(
                reqs_to_load={},
                reqs_to_store={},
                reqs_to_flush={req_key},
            )
            worker_connector.bind_connector_metadata(flush_metadata)
            worker_connector.handle_preemptions(flush_metadata)
            worker_connector.start_load_kv(dummy_ctx)
            worker_connector.clear_connector_metadata()

            finished_sending, finished_recving = worker_connector.get_finished(set())
            finished_sending_all |= finished_sending
            finished_recving_all |= finished_recving

            pending_store_jobs = self._get_pending_store_jobs(worker_connector, req_key)
            if pending_store_jobs:
                flush_deadline = time.time() + 30.0
                while pending_store_jobs and time.time() < flush_deadline:
                    time.sleep(0.01)
                    finished_sending, finished_recving = worker_connector.get_finished(
                        set()
                    )
                    finished_sending_all |= finished_sending
                    finished_recving_all |= finished_recving
                    pending_store_jobs = self._get_pending_store_jobs(
                        worker_connector, req_key
                    )

                finished_sending, finished_recving = worker_connector.get_finished(
                    {req_key}
                )
                finished_sending_all |= finished_sending
                finished_recving_all |= finished_recving

                if req_key not in finished_sending_all:
                    raise RuntimeError(
                        f"timed out waiting for deferred KV store flush for {req_key}"
                    )

            connector_stats_summary = self._collect_connector_stats(worker_connector)

        except Exception as exc:
            connector_stats_summary = self._collect_connector_stats(worker_connector)
            return ScenarioRequestResult(
                request_id=spec.request_id,
                doc_tokens=spec.doc_tokens,
                reuse_source_id=spec.reuse_source_id,
                reuse_prefix_len=spec.reuse_prefix_len,
                logical_request_id=spec.logical_request_id,
                prompt_key=spec.prompt_key,
                traffic_scope=spec.traffic_scope,
                target_instance=spec.target_instance,
                scheduler_steps=scheduler_steps,
                total_scheduled_tokens=total_scheduled_tokens,
                max_scheduled_tokens_per_step=max_scheduled_tokens_per_step,
                connector_load_ops=int(connector_stats_summary["connector_load_ops"]),
                connector_load_bytes=int(
                    connector_stats_summary["connector_load_bytes"]
                ),
                connector_load_time_s=float(
                    connector_stats_summary["connector_load_time_s"]
                ),
                connector_store_ops=int(connector_stats_summary["connector_store_ops"]),
                connector_store_bytes=int(
                    connector_stats_summary["connector_store_bytes"]
                ),
                connector_store_time_s=float(
                    connector_stats_summary["connector_store_time_s"]
                ),
                finished_sending=False,
                finished_recving=False,
                successful=False,
                duration_s=time.time() - start,
                error=str(exc),
            )

        return ScenarioRequestResult(
            request_id=spec.request_id,
            doc_tokens=spec.doc_tokens,
            reuse_source_id=spec.reuse_source_id,
            reuse_prefix_len=spec.reuse_prefix_len,
            logical_request_id=spec.logical_request_id,
            prompt_key=spec.prompt_key,
            traffic_scope=spec.traffic_scope,
            target_instance=spec.target_instance,
            scheduler_steps=scheduler_steps,
            total_scheduled_tokens=total_scheduled_tokens,
            max_scheduled_tokens_per_step=max_scheduled_tokens_per_step,
            connector_load_ops=int(connector_stats_summary["connector_load_ops"]),
            connector_load_bytes=int(connector_stats_summary["connector_load_bytes"]),
            connector_load_time_s=float(
                connector_stats_summary["connector_load_time_s"]
            ),
            connector_store_ops=int(connector_stats_summary["connector_store_ops"]),
            connector_store_bytes=int(connector_stats_summary["connector_store_bytes"]),
            connector_store_time_s=float(
                connector_stats_summary["connector_store_time_s"]
            ),
            finished_sending=req_key in finished_sending_all,
            finished_recving=req_key in finished_recving_all,
            successful=True,
            duration_s=time.time() - start,
            error=None,
        )

    def run_requests(
        self, specs: list[ScenarioRequestSpec]
    ) -> list[ScenarioRequestResult]:
        return [self.run_request(spec) for spec in specs]

    def shutdown(self) -> None:
        if self._runtime is None:
            return
        scheduler_connector = self._runtime.get("scheduler_connector")
        worker_connector = self._runtime.get("worker_connector")
        if worker_connector is not None:
            worker_connector.shutdown()
        if scheduler_connector is not None:
            scheduler_connector.shutdown()
