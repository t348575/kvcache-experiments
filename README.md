<h1 align="center">kvcache-experiments</h1>
<p align="center">
  <img alt="Views" src="https://lambda.348575.xyz/repo-view-counter?repo=kvcache-experiments"/>
</p>

<p align="center">
  Collection of benchmarks, benchmark drivers, harnesses, plotting, and various scripts for KV cache offloading in vLLM. Drives vLLM servers through repeatable workloads, records time to first token (TTFT) as well as various other statistics. Designed to compare <a href="https://github.com/t348575/py-kvcache/">py-kvcache</a>, <a href="https://github.com/lmcache/lmcache">lmcache</a>, <a href="https://github.com/vllm-project/vllm">vllm</a> kv offload, <a href="https://github.com/llm-d/llm-d-kv-cache/">llm-d</a>.
</p>

Expected to be run with:

- The vLLM fork [t348575/vllm](https://github.com/t348575/vllm) but not required.
- The profiler [t348575/simple-profiler](https://github.com/t348575/simple-profiler/), required for some analysis scripts.
- A virtualenv with `vLLM` and other script dependencies.

Gated models such as Llama require `HF_TOKEN` in the environment. Run every script from the repository root so the `common/` package imports resolve.

## Main scripts

### bench.py

A Benchmark runner that reads a JSON config. For each entry it starts a vLLM server, waits for health, runs a benchmark driver, then shuts the server down and writes one row per run to a timestamped CSV.

```bash
python bench.py --config bench_configs/baseline.json
```

This script orchestrates and starts `vLLM` with one of `N` specified configurations (i.e. different vLLM configurations), checks its health then runs one of the available benchmarks: [prefix_cache_benchmark](./prefix_cache_benchmark.py), [sharegpt](https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered), [bailian](https://github.com/alibaba-edu/qwen-bailian-usagetraces-anon) (alibaba qwen traces), [LongBench v2](https://huggingface.co/datasets/zai-org/LongBench-v2), [SCBench](https://huggingface.co/datasets/microsoft/SCBench).

| Flag | Meaning |
| --- | --- |
| `--config` | Path to the JSON config (default `bench_config.json`). |
| `--resume` | Path to a prior results CSV. Allows resuming or re-running failed runs. |
| `--data-dir` | Base directory substituted for the `{data_dir}` token in storage paths, overrides the config's top-level `data_dir`. Useful when running benchmarks on slurm or similar systems, where local scratch directory names change. |

Various bench configuration presets already tested are present in [bench_configs](./bench_configs/)

### scripts/pareto_measure.py

This script is essential for understanding when KV caching is worth it. This script is used to determine the break even point for the pareto break even feature present in [py-kvcache](https://github.com/t348575/py-kvcache).

The two TTFT curves that define the KV cache Pareto frontier:

- `cold_prefill` f(N): TTFT to compute N tokens with prefix caching off.
- `cache_hit` g(P): TTFT when P tokens load from a cache plus a one output token.

The frontier in (doc_size × prefix_fraction) space sits where `f(D) = g(frac·D) + f((1−frac)·D)`.

[pareto_config.json](./pareto_config.json) is the default configuration used for capturing the pareto frontier. 

```bash
python -m scripts.pareto_measure --config pareto_config.json
```

| Flag | Meaning |
| --- | --- |
| `--config` | Path to the pareto JSON config (default `pareto_config.json`). |
| `--server-configs` | Subset of server configs to run (default: all in the config). |
| `--wipe-shared-storage` | After each storage job, `rm -rf` its `shared_storage_path`. |
| `--data-dir` | Base directory substituted for the `{data_dir}` token in storage paths (required). |

#### Break even json file

Feed the result CSV to [plots/pareto_plot.py](./plots/pareto_plot.py) for the frontier plot, and to [scripts/emit_break_even.py](./scripts/emit_break_even.py) to generate the gating data py-kvcache consumes.

### prefix_cache_benchmark.py

The request driver `bench.py` and `pareto_measure.py` call. It sends a schedule of requests to a running vLLM OpenAI endpoint where a chosen fraction reuse an earlier request's prefix, then reports TTFT for fresh vs reuse and the speedup between them. This is largely based on the lmcache long doc benchmark & the vLLM long doc online benchmark.

```bash
python prefix_cache_benchmark.py \
  --port 8000 --num-requests 48 --doc-size 10240 \
  --prefix-reuse-pct 0.5 --prefix-size 1.0 \
  --max-concurrency 10 --output-len 1 --csv-output run.csv
```

| Flag | Meaning |
| --- | --- |
| `--num-requests` | Total requests to send (required). |
| `--doc-size` | Document size in tokens: `10240` or a range `10240-40960` (required). |
| `--prefix-reuse-pct` | Fraction of requests that reuse a prior prefix (0.0–1.0). |
| `--prefix-size` | Reused prefix as a fraction of the source doc: `0.5` or a range. |
| `--max-concurrency` | Max in-flight requests. |
| `--arrival-rate` | Poisson arrival rate in req/s; omit to send as fast as concurrency allows. |
| `--pre-warmup-requests` | Untimed warmup requests sent before measurement (default 5). These are necessary, since the first 1-3 requests usually have very high latency. |
| `--csv-output` / `--json-output` | Write per-request CSV / print a JSON summary line. |

## Other scripts
- [scripts](./scripts/) contains various utility and smaller benchmark scripts, dataset fetch scripts, etc.
  - [scripts/bailian_replay.py](./scripts/bailian_replay.py) for playing the alibaba qwen traces.
  - [scripts/longbench_v2_replay.py](./scripts/longbench_v2_replay.py) for playing the LongBench v2 data. This uses the following format: send the base dataset document, then send N followup requests with randomly generated data appended to the end.
  - [scripts/scbench_replay.py](./scripts/scbench_replay.py) replicates the SCBench multi-turn benchmark flow.
  - Various other scripts for standalone kv benchmarks, FS benchmarks, etc.
- [plots](./plots/) various plotting scripts to visualize the benchmark data.
