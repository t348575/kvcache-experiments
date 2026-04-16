#!/usr/bin/env python3
from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass
from typing import Any

SYNTHETIC_NUM_KV_HEADS = 1
SYNTHETIC_HEAD_SIZE = 128
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
    block_size: int = 16
    num_blocks: int = 4096
    max_num_batched_tokens: int = MAX_NUM_BATCHED_TOKENS
    max_num_seqs: int = MAX_NUM_SEQS
    enable_chunked_prefill: bool = True
    enable_prefix_caching: bool = True
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


@dataclass
class ScenarioRequestResult:
    request_id: int
    doc_tokens: int
    reuse_source_id: int | None
    reuse_prefix_len: int
    finished_sending: bool
    finished_recving: bool
    successful: bool
    duration_s: float
    error: str | None = None


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
) -> list[ScenarioRequestSpec]:
    if not 0.0 <= store_pct <= 1.0:
        raise ValueError("store_pct must be in [0.0, 1.0]")

    rng = random.Random(seed)
    requests: list[ScenarioRequestSpec] = []
    stored: list[ScenarioRequestSpec] = []

    for request_id in range(num_requests):
        doc_tokens = rng.randint(*doc_size_range)
        should_store = not stored or rng.random() < store_pct
        if should_store:
            prompt_token_ids = tuple(
                _unique_token(request_id, token_idx) for token_idx in range(doc_tokens)
            )
            request = ScenarioRequestSpec(
                request_id=request_id,
                doc_tokens=doc_tokens,
                reuse_source_id=None,
                reuse_prefix_len=0,
                prompt_token_ids=prompt_token_ids,
            )
            requests.append(request)
            stored.append(request)
            continue

        source = rng.choice(stored)
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
            )
        )

    return requests


def results_to_rows(results: list[ScenarioRequestResult]) -> list[dict[str, Any]]:
    return [asdict(result) for result in results]


def _unique_token(request_id: int, token_idx: int) -> int:
    return request_id * 1_000_000 + token_idx + 1


class RuntimeKVConnectorHarness:
    def __init__(self, config: RuntimeConnectorConfig):
        self.config = config
        self._runtime: dict[str, Any] | None = None

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
                ModelConfig,
                SchedulerConfig,
                VllmConfig,
                set_current_vllm_config,
            )
            from vllm.distributed.kv_transfer.kv_connector.factory import (
                KVConnectorFactory,
            )
            from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
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

        device = torch.device(DEVICE)

        model_config = ModelConfig(
            model=self.config.model_name,
            trust_remote_code=True,
            dtype=MODEL_DTYPE,
            seed=42,
            hf_overrides={},
        )
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
            kv_connector_extra_config=self.config.connector_extra_config or {},
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

        kv_cache_spec = FullAttentionSpec(
            block_size=self.config.block_size,
            num_kv_heads=SYNTHETIC_NUM_KV_HEADS,
            head_size=SYNTHETIC_HEAD_SIZE,
            dtype=dtype,
        )
        kv_cache_config = KVCacheConfig(
            num_blocks=self.config.num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=kv_cache_spec.page_size_bytes * self.config.num_blocks,
                    shared_by=["layer0"],
                )
            ],
            kv_cache_groups=[KVCacheGroupSpec(["layer0"], kv_cache_spec)],
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

        if cache_mode == "cross_layer":
            module_name, _, class_name = CROSS_LAYER_BACKEND_PATH.partition(":")
            backend_module = importlib.import_module(module_name)
            backend_cls = getattr(backend_module, class_name)
            with set_current_vllm_config(vllm_config):
                _, cross_layers_kv_cache, _ = (
                    KVConnectorModelRunnerMixin.allocate_uniform_kv_caches(
                        kv_cache_config=kv_cache_config,
                        attn_groups=[
                            [
                                AttentionGroup(
                                    backend=backend_cls,
                                    layer_names=["layer0"],
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
            layer_tensor = torch.zeros(
                (
                    2,
                    self.config.num_blocks,
                    self.config.block_size,
                    SYNTHETIC_NUM_KV_HEADS,
                    SYNTHETIC_HEAD_SIZE,
                ),
                dtype=dtype,
                device=device,
            )
            worker_connector.register_kv_caches({"layer0": layer_tensor})
        else:
            raise ValueError(
                "register_cache_mode must be one of: auto, cross_layer, layer_dict"
            )

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
            "Request": Request,
            "get_request_block_hasher": get_request_block_hasher,
            "sha256": sha256,
            "scheduler": scheduler,
            "scheduler_connector": scheduler_connector,
            "worker_connector": worker_connector,
            "dummy_ctx": dummy_ctx,
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
            sampled_token_ids=[[0] for _ in req_ids],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=None,
            kv_connector_output=kv_connector_output,
        )

    def run_request(self, spec: ScenarioRequestSpec) -> ScenarioRequestResult:
        runtime = self._ensure_runtime()
        scheduler = runtime["scheduler"]
        worker_connector = runtime["worker_connector"]
        dummy_ctx = runtime["dummy_ctx"]
        EMPTY_MODEL_RUNNER_OUTPUT = runtime["EMPTY_MODEL_RUNNER_OUTPUT"]
        KVConnectorOutput = runtime["KVConnectorOutput"]

        request = self._make_request(spec)
        finished_sending_all: set[str] = set()
        finished_recving_all: set[str] = set()
        start = time.time()

        try:
            scheduler.add_request(request)

            while scheduler.requests:
                scheduler_output = scheduler.schedule()
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

            while scheduler.requests:
                scheduler_output = scheduler.schedule()
                finished_sending, finished_recving = worker_connector.get_finished(
                    scheduler_output.finished_req_ids
                )
                finished_sending_all |= finished_sending
                finished_recving_all |= finished_recving
                model_runner_output = EMPTY_MODEL_RUNNER_OUTPUT
                model_runner_output.kv_connector_output = KVConnectorOutput(
                    finished_sending=finished_sending,
                    finished_recving=finished_recving,
                )
                scheduler.update_from_output(scheduler_output, model_runner_output)

        except Exception as exc:
            return ScenarioRequestResult(
                request_id=spec.request_id,
                doc_tokens=spec.doc_tokens,
                reuse_source_id=spec.reuse_source_id,
                reuse_prefix_len=spec.reuse_prefix_len,
                finished_sending=False,
                finished_recving=False,
                successful=False,
                duration_s=time.time() - start,
                error=str(exc),
            )

        req_key = request.request_id
        return ScenarioRequestResult(
            request_id=spec.request_id,
            doc_tokens=spec.doc_tokens,
            reuse_source_id=spec.reuse_source_id,
            reuse_prefix_len=spec.reuse_prefix_len,
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
