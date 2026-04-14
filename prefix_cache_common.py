#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass

from openai import AsyncOpenAI


@dataclass
class RequestSpec:
    request_id: int
    doc_tokens: int
    reuse_source_id: int | None = None
    reuse_prefix_len: int | None = None
    scheduled_time: float = 0.0


@dataclass
class RequestResult:
    request_id: int
    doc_tokens: int
    is_prefix_reuse: bool
    reuse_prefix_len: int
    scheduled_time: float
    request_start: float
    ttft: float
    request_end: float
    successful: bool


def parse_int_range(value: str) -> tuple[int, int]:
    if "-" in value:
        parts = value.split("-", 1)
        lo, hi = int(parts[0]), int(parts[1])
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    n = int(value)
    return n, n


def parse_pct_range(value: str) -> tuple[float, float]:
    if "-" in value:
        parts = value.split("-", 1)
        lo, hi = float(parts[0]), float(parts[1])
        if lo > hi:
            lo, hi = hi, lo
        return lo, hi
    p = float(value)
    return p, p


def generate_request_schedule(
    num_requests: int,
    doc_size_lo: int,
    doc_size_hi: int,
    prefix_reuse_pct: float,
    prefix_pct_lo: float,
    prefix_pct_hi: float,
    arrival_rate: float | None,
    rng: random.Random,
) -> list[RequestSpec]:
    specs: list[RequestSpec] = []
    elapsed = 0.0

    for i in range(num_requests):
        doc_tokens = rng.randint(doc_size_lo, doc_size_hi)
        spec = RequestSpec(request_id=i, doc_tokens=doc_tokens)

        if i > 0 and rng.random() < prefix_reuse_pct:
            source = rng.choice(specs)
            frac = rng.uniform(prefix_pct_lo, prefix_pct_hi)
            prefix_len = max(1, int(frac * source.doc_tokens))
            spec.reuse_source_id = source.request_id
            spec.reuse_prefix_len = prefix_len

        if arrival_rate is not None and arrival_rate > 0:
            elapsed += rng.expovariate(arrival_rate)
        spec.scheduled_time = elapsed
        specs.append(spec)

    return specs


def _build_document(request_id: int, num_tokens: int) -> str:
    return f"{request_id} " + " ".join(["hi"] * num_tokens)


def build_prompt(spec: RequestSpec, prior_docs: dict[int, str]) -> str:
    if spec.reuse_source_id is not None and spec.reuse_source_id in prior_docs:
        source_doc = prior_docs[spec.reuse_source_id]
        words = source_doc.split()
        prefix_words = words[: spec.reuse_prefix_len]
        remaining = max(0, spec.doc_tokens - len(prefix_words))
        suffix = [f"x{spec.request_id}"] + ["hi"] * remaining
        return " ".join(prefix_words + suffix)

    doc = _build_document(spec.request_id, spec.doc_tokens)
    prior_docs[spec.request_id] = doc
    return doc


def build_prompts(specs: list[RequestSpec]) -> list[str]:
    prior_docs: dict[int, str] = {}
    return [build_prompt(spec, prior_docs) for spec in specs]


def _has_content(chunk, completions_mode: bool) -> bool:
    if not chunk.choices:
        return False
    if completions_mode:
        return chunk.choices[0].text is not None
    delta = chunk.choices[0].delta
    if delta is None:
        return False
    if hasattr(delta, "content") and delta.content:
        return True
    for key in ("reasoning_content", "reasoning"):
        if hasattr(delta, key) and getattr(delta, key):
            return True
    return False


def _extract_content(chunk, completions_mode: bool) -> str:
    if completions_mode:
        return chunk.choices[0].text or ""
    delta = chunk.choices[0].delta
    if hasattr(delta, "content") and delta.content:
        return delta.content
    for key in ("reasoning_content", "reasoning"):
        if hasattr(delta, key) and getattr(delta, key):
            return getattr(delta, key)
    return ""


async def send_request(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    spec: RequestSpec,
    output_len: int,
    completions_mode: bool,
    eos_token_id: int | None,
    benchmark_start: float,
) -> RequestResult:
    now = time.time()
    wait = benchmark_start + spec.scheduled_time - now
    if wait > 0:
        await asyncio.sleep(wait)

    start_time = time.time()
    first_token_time = None
    logit_bias = {str(eos_token_id): -100} if eos_token_id is not None else None

    if completions_mode:
        stream = await client.completions.create(
            model=model,
            prompt=prompt,
            stream=True,
            max_tokens=output_len,
            temperature=0.0,
            stream_options={"include_usage": True},
            logit_bias=logit_bias,
        )
    else:
        stream = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            max_tokens=output_len,
            temperature=0.0,
            stream_options={"include_usage": True},
            logit_bias=logit_bias,
        )

    async for chunk in stream:
        if _has_content(chunk, completions_mode):
            content = _extract_content(chunk, completions_mode)
            if first_token_time is None and content:
                first_token_time = time.time()

    end_time = time.time()
    ttft = (first_token_time - start_time) if first_token_time is not None else -1.0

    return RequestResult(
        request_id=spec.request_id,
        doc_tokens=spec.doc_tokens,
        is_prefix_reuse=spec.reuse_source_id is not None,
        reuse_prefix_len=spec.reuse_prefix_len or 0,
        scheduled_time=spec.scheduled_time,
        request_start=start_time,
        ttft=ttft,
        request_end=end_time,
        successful=ttft > 0,
    )


async def run_benchmark(
    client: AsyncOpenAI,
    model: str,
    specs: list[RequestSpec],
    output_len: int,
    max_concurrency: int,
    completions_mode: bool,
    eos_token_id: int | None,
    prompts: list[str] | None = None,
) -> list[RequestResult]:
    semaphore = asyncio.Semaphore(max_concurrency)
    benchmark_prompts = prompts if prompts is not None else build_prompts(specs)
    benchmark_start = time.time()

    async def _guarded(spec: RequestSpec, prompt: str) -> RequestResult:
        async with semaphore:
            return await send_request(
                client,
                model,
                prompt,
                spec,
                output_len,
                completions_mode,
                eos_token_id,
                benchmark_start,
            )

    tasks = [_guarded(spec, prompt) for spec, prompt in zip(specs, benchmark_prompts)]
    results = await asyncio.gather(*tasks)
    return list(results)
