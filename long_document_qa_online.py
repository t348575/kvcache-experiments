from dataclasses import dataclass
import asyncio
import argparse
import openai
import datetime
import time
import requests
import sys

from vllm.transformers_utils.tokenizer import get_tokenizer
from prometheus_client.parser import text_string_to_metric_families

import dataset


@dataclass
class VLLM_METRICS:
    prefix_cache_quires: int
    prefix_cache_hits: int
    prompt_tokens: int


@dataclass
class Response:
    send_time: float
    first_token_time: float
    last_token_time: float
    prompt_tokens_num: int
    completion_tokens_num: int
    generation_text: str


def get_vllm_metrics(metrics_url):
    """
    Get the vLLM metrics via the /metrics endpoint.

    Args:
        metrics_url (str): The URL of the metrics endpoint.

    Returns:
        _type_: _description_
    """
    resp = requests.get(metrics_url)
    resp.raise_for_status()
    metrics_text = resp.text

    # for cur_metric in text_string_to_metric_families(metrics_text):
    #     print(cur_metric.name)

    prefix_cache_queries = 0
    prefix_cache_hits = 0
    prompt_tokens = 0
    for cur_metric in text_string_to_metric_families(metrics_text):
        # v1 often exposes counters like these:
        if cur_metric.name == "vllm:prefix_cache_queries":
            for cur_sample in cur_metric.samples:
                prefix_cache_queries += cur_sample.value
        if cur_metric.name == "vllm:prefix_cache_hits":
            for sample in cur_metric.samples:
                prefix_cache_hits += sample.value
        if cur_metric.name == "vllm:request_prompt_tokens":
            for sample in cur_metric.samples:
                prompt_tokens += sample.value
        if cur_metric.name == "vllm:request_queue_time_seconds":
            for sample in cur_metric.samples:
                # print(f"Request queue time: {sample.value} seconds")
                pass

    return VLLM_METRICS(
        prefix_cache_quires=prefix_cache_queries,
        prefix_cache_hits=prefix_cache_hits,
        prompt_tokens=prompt_tokens,
    )


def compute_vllm_metrics_statistics(start_metrics, end_metrics):
    """
    Compute the vLLM metrics statistics between two time points.

    Args:
        start_metrics (VLLM_METRICS): The metrics at the start time.
        end_metrics (VLLM_METRICS): The metrics at the end time.

    Returns:
        dict: A dictionary containing the computed statistics.
    """
    num_prompt_tokens = end_metrics.prompt_tokens - start_metrics.prompt_tokens
    num_prefix_hit_tokens = (
        end_metrics.prefix_cache_hits - start_metrics.prefix_cache_hits
    )

    prefix_cache_hit_rate = num_prefix_hit_tokens / num_prompt_tokens

    return {
        "total_prompt_tokens": num_prompt_tokens,
        "total_cache_hits": num_prefix_hit_tokens,
        "hit_rate": prefix_cache_hit_rate,
    }


def aggregate_statistics(responses):
    """
    Aggregate statistics from the responses.

    Args:
        responses (list[Response]): List of response objects.

    Returns:
        dict: A dictionary containing the aggregated statistics.
    """
    total_requests = len(responses)
    average_ttft = (
        sum((r.first_token_time - r.send_time) for r in responses if r.first_token_time)
        / total_requests
    )

    return {
        "total_requests": total_requests,
        "average_ttft": average_ttft,
    }


async def process_single_prompt(client, model, prompt, output_len):
    """
    Submit a single prompt in a streaming manner.

    Args:
        client (openai.AsyncOpenAI): The OpenAI API client.
        model (str): The model to use for the completion.
        prompt (str): The prompt to send to the model.
        output_len (int): The maximum length of the output.

    Returns:
        Response: The response object containing the completion details.
    """
    first_token_time = None
    last_token_time = None
    send_time = time.perf_counter()
    token_stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        stream=True,
        max_tokens=output_len,
        stream_options={"include_usage": True},
    )

    generation_text = ""
    # event is: check_completion_chunk https://platform.openai.com/docs/api-reference/chat_streaming/streaming
    async for event in token_stream:
        choices = getattr(event, "choices", None)
        if choices == None or len(choices) == 0:
            continue

        delta = getattr(choices[0], "delta", None)
        if delta == None:
            continue

        gen_tokens = getattr(choices[0].delta, "content", None)
        if gen_tokens == None or gen_tokens == "":
            continue

        if first_token_time == None:
            first_token_time = time.perf_counter()

        generation_text += gen_tokens

    if first_token_time is None:
        # In some cases when the output_len is 1 and the first output token is a special token
        # It will send an event, so the first_token_time will not be set
        first_token_time = time.perf_counter()
    last_token_time = time.perf_counter()
    prompt_tokens_num = event.usage.prompt_tokens
    completion_tokens_num = event.usage.completion_tokens

    return Response(
        send_time=send_time,
        first_token_time=first_token_time,
        last_token_time=last_token_time,
        prompt_tokens_num=prompt_tokens_num,
        completion_tokens_num=completion_tokens_num,
        generation_text=generation_text,
    )


async def run_long_document_qa(
    client, model, prompt_dataset, output_len, max_concurrent_tasks
):
    """
    Run the long document QA task with the given dataset.

    Args:
        client (openai.AsyncOpenAI): The openai client.
        model (str): The model to use for the completion.
        prompt_dataset (dataset.Dataset): The dataset containing prompts.
        output_len (int): The maximum length of the output.

    Returns:
        list[Response]: A list of response objects containing the completion details.
    """
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    results = []

    async def sem_task(prompt):
        async with semaphore:
            return await process_single_prompt(client, model, prompt, output_len)

    tasks = [
        asyncio.create_task(sem_task(prompt_dataset.next_item()))
        for _ in range(prompt_dataset.size())
    ]

    for task in asyncio.as_completed(tasks):
        result = await task
        results.append(result)

    return results


async def run_long_document_qa_with_interval(
    client, model, prompt_dataset, output_len, interval, max_concurrent_tasks
):
    """
    Run the long document QA task with the given dataset and rate limit.

    Args:
        client (openai.AsyncOpenAI): The openai client.
        model (str): The model to use for the completion.
        prompt_dataset (dataset.Dataset): The dataset containing prompts.
        output_len (int): The maximum length of the output.
        rate (float): The rate limit in requests per second.

    Returns:
        list[Response]: A list of response objects containing the completion details.
    """
    raise NotImplementedError("This function is not implemented yet.")
    tasks = []

    for i in range(prompt_dataset.size()):
        prompt = prompt_dataset.next_item()
        task = asyncio.create_task(
            process_single_prompt(client, model, prompt, output_len)
        )
        tasks.append(task)
        # print(
        #     f"Submitted request {i+1}/{prompt_dataset.size()} at {time.perf_counter()}"
        # )
        await asyncio.sleep(interval)

    results = await asyncio.gather(*tasks)

    return results


async def main(args):
    client = openai.AsyncOpenAI(api_key=args.api_key, base_url=args.base_url)
    model = args.model

    max_ongoing_tasks = args.max_ongoing_requests

    if args.output_file == "":
        output_f = sys.stdout
    else:
        output_f = args.output_file

    interval = 0
    if args.intensity != 0:
        interval = 1.0 / args.intensity

    # create dataset, the warm-up dataset will generate the KV cache
    # NOTE: the warm dataset will have the same prefixes as the test dataset
    tokenizer = get_tokenizer(model, trust_remote_code=True)
    test_dataset = dataset.RandomizedPrefixDataset(
        tokenizer=tokenizer,
        document_length=args.document_length,
        num_documents=args.num_documents,
        document_repeat=args.document_repeat,
        document_repeat_mode=args.document_repeat_mode,
        shuffle_seed=args.shuffle_seed,
        max_entries=args.max_requests,
    )

    if args.warmup_mode == "prepare-prefix":
        # use the deduplicated dataset for warm up
        warmup_dataset = test_dataset.get_dedup_dataset()
    elif args.warmup_mode == "random":
        # use a separate warm up dataset
        if args.warmup_docs == 0:
            args.warmup_docs = args.num_documents
        if args.warmup_document_length == 0:
            args.warmup_document_length = args.document_length
        warmup_dataset = dataset.RandomizedPrefixDataset(
            tokenizer=tokenizer,
            document_length=args.warmup_document_length,
            num_documents=args.warmup_docs,
            document_repeat=1,
            document_repeat_mode="min-distance",
            shuffle_seed=args.shuffle_seed,
        )
    else:
        raise ValueError(f"Unsupported warmup mode: {args.warmup_mode}")

    if args.base_url.endswith("/v1/"):
        metrics_url = args.base_url[:-4] + "/metrics"
    else:
        metrics_url = args.base_url + "/metrics"

    # Warm up
    print("------warm up------")
    warmup_start_time = time.perf_counter()
    warmup_results = await run_long_document_qa(
        client=client,
        model=model,
        prompt_dataset=warmup_dataset,
        output_len=args.output_len,
        max_concurrent_tasks=max_ongoing_tasks,
    )
    warmup_end_time = time.perf_counter()
    warmup_time = warmup_end_time - warmup_start_time
    print(f"Warm up time: {warmup_time:.2f} seconds")
    print("------warm up ends------")

    # Flush
    if args.enable_flush:
        flush_dataset = dataset.RandomizedPrefixDataset(
            tokenizer=tokenizer,
            document_length=args.flush_document_length,
            num_documents=args.num_flush_docs,
            document_repeat=1,
            document_repeat_mode="min-distance",
            shuffle_seed=args.shuffle_seed,
        )
        print("------flush GPU and CPU cache------")
        flush_start_time = time.perf_counter()
        flush_results = await run_long_document_qa(
            client=client,
            model=model,
            prompt_dataset=flush_dataset,
            output_len=args.output_len,
            max_concurrent_tasks=max_ongoing_tasks,
        )
        flush_end_time = time.perf_counter()
        flush_time = flush_end_time - flush_start_time
        print(f"Flush time: {flush_time:.2f} seconds")

    # Test
    print("------test------")
    start_metrics = get_vllm_metrics(metrics_url)
    test_start_time = time.perf_counter()
    if interval == 0:
        results = await run_long_document_qa(
            client=client,
            model=model,
            prompt_dataset=test_dataset,
            output_len=args.output_len,
            max_concurrent_tasks=max_ongoing_tasks,
        )
    else:
        results = await run_long_document_qa_with_interval(
            client=client,
            model=model,
            prompt_dataset=test_dataset,
            output_len=args.output_len,
            interval=interval,
            max_concurrent_tasks=max_ongoing_tasks,
        )
    test_end_time = time.perf_counter()
    total_time = test_end_time - test_start_time
    end_metrics = get_vllm_metrics(metrics_url)

    # TODO: We want: ttft, total run time, cache hit rate
    print(f"Benchmark results: ", file=output_f)
    print(f"  Total time: {total_time:.2f} seconds", file=output_f)

    statistics = aggregate_statistics(results)
    for item in results:
        print(f"Prompt tokens: {item.prompt_tokens_num}, output tokens: {item.completion_tokens_num}")
    print(f"  Total requests: {statistics['total_requests']}", file=output_f)
    print(
        f"  Average time to first token: {statistics['average_ttft']:.2f} seconds",
        file=output_f,
    )

    vllm_statistics = compute_vllm_metrics_statistics(start_metrics, end_metrics)
    print(
        f"  Total prompt tokens: {vllm_statistics['total_prompt_tokens']}",
        file=output_f,
    )
    print(f"  Total cache hits: {vllm_statistics['total_cache_hits']}", file=output_f)
    print(f"  Cache hit rate: {vllm_statistics['hit_rate']*100:.2f}%", file=output_f)

    # Save detailed output to file
    if args.detail_output_file != "":
        detailed_f = open(args.detail_output_file, "w")
        for cur_result in results:
            print("-----", file=detailed_f)
            print(f"Prompt tokens: {cur_result.prompt_tokens_num}", file=detailed_f)
            print(
                f"Completion tokens: {cur_result.completion_tokens_num}",
                file=detailed_f,
            )
            print(f"Send time: {cur_result.send_time}", file=detailed_f)
            print(f"First token time: {cur_result.first_token_time}", file=detailed_f)
            print(f"Last token time: {cur_result.last_token_time}", file=detailed_f)
            print(f"TTFT: {cur_result.first_token_time - cur_result.send_time}", file=detailed_f)
            print(f"Generation text: {cur_result.generation_text}", file=detailed_f)


def create_argument_parser():
    parser = argparse.ArgumentParser(
        description="Benchmark the performance with or "
        "without automatic prefix caching."
    )

    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen3-0.6B", help="Model name"
    )

    parser.add_argument(
        "--max-requests",
        type=int,
        default=None,
        help="Number of total requests, unset means unlimited",
    )

    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:8000/v1/",
        help="vLLM server base url",
    )

    parser.add_argument(
        "--intensity",
        type=float,
        default=0,
        help="Intensity of the workload, num_req/sec",
    )

    parser.add_argument(
        "--api-key", type=str, default="EMPTY", help="vLLM server api key, not used"
    )

    parser.add_argument(
        "--max-ongoing-requests",
        type=int,
        default=20,
        help="Maximum number of ongoing requests",
    )

    parser.add_argument(
        "--document-length",
        type=int,
        default=20000,
        help="Range of input lengths for sampling prompts, "
        'specified as "min:max" (e.g., "128:256").',
    )

    parser.add_argument(
        "--num-documents",
        type=int,
        default=8,
        help="Range of input lengths for sampling prompts, "
        'specified as "min:max" (e.g., "128:256").',
    )

    parser.add_argument(
        "--document-repeat",
        type=int,
        default=1,
        help="How many times each prompt occurs",
    )
    parser.add_argument(
        "--document-repeat-mode",
        type=str,
        default="random",
        help="How to repeat the prompts",
    )

    parser.add_argument("--output-len", type=int, default=1)

    parser.add_argument(
        "--shuffle-seed",
        type=int,
        default=0,
        help='Random seed when the repeat mode is "random"',
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="",
        help="Path to the output file, unset = stdtout",
    )

    parser.add_argument(
        "--detail-output-file",
        type=str,
        default="",
        help="Path to the detailed output file, unset = no detailed output",
    )

    parser.add_argument(
        "--log",
        type=str,
        default="",
        help="Path to the log file, unset = stdout",
    )

    # warm up settings
    parser.add_argument(
        "--warmup-mode",
        type=str,
        default="prepare-prefix",
        help="warm up mode, random or prepare-prefix",
    )

    parser.add_argument(
        "--warmup-docs",
        type=int,
        default=0,
        help="Number of warm up docs",
    )

    parser.add_argument(
        "--warmup-document-length",
        type=int,
        default=0,
        help="Length of each warm up doc",
    )

    # flush settings
    parser.add_argument(
        "--enable-flush",
        action="store_true",
        help="Whether to enable flushing the cache before the test",
    )
    parser.add_argument(
        "--num-flush-docs",
        type=int,
        default=2,
        help="Number of flush docs",
    )
    parser.add_argument(
        "--flush-document-length",
        type=int,
        default=90000,
        help="Length of each flush doc",
    )

    return parser


def print_arguments(args):
    print("Experiment Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    print("Experiment Configuration ends")


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()

    print("Experiment start at: ", datetime.datetime.now())
    print_arguments(args)

    asyncio.run(main(args))

    print("Experiment end at: ", datetime.datetime.now())

"""
Example command:
python long_document_qa_online.py --base_url http://localhost:8000/v1/ --model Qwen/Qwen3-0.6B --document-length 20000 --num-documents 8 --document-repeat 5 --document-repeat-mode random --output-len 10 
"""
