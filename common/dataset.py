import hashlib
import random
import json
from dataclasses import dataclass, field

CACHE_DIR = "./cached_data"

# Bailian/Qwen traces group tokens into fixed-size blocks; one hash id per block.
QWEN_TOKENS_PER_BLOCK = 16

# Maps the short task name to the on-disk trace filename stem. The files are not
# uniformly "qwen_trace<task>" (coder/thinking drop the "trace" infix), so an
# explicit map avoids constructing wrong paths.
QWEN_TRACE_FILES = {
    "A": "qwen_traceA_blksz_16",
    "B": "qwen_traceB_blksz_16",
    "coder": "qwen_coder_blksz_16",
    "thinking": "qwen_thinking_blksz_16",
}

# SCBench (microsoft/SCBench) is a shared-context, multi-turn benchmark: each row
# is one session = a long `context` plus a list of `multi_turns` follow-ups that
# all reuse that context. Unlike the pre-tokenized Bailian traces, prefix reuse
# here is structural (the context is the shared prefix) and token counts depend
# on the tokenizer, so they are computed at load time rather than baked in.
SCBENCH_REPO = "microsoft/SCBench"
SCBENCH_PARQUET_GLOB = "{config}/test-*.parquet"
SCBENCH_CONFIGS = [
    "scbench_choice_eng",
    "scbench_kv",
    "scbench_many_shot",
    "scbench_mf",
    "scbench_prefix_suffix",
    "scbench_qa_chn",
    "scbench_qa_eng",
    "scbench_repoqa",
    "scbench_repoqa_and_kv",
    "scbench_summary",
    "scbench_summary_with_needles",
    "scbench_vt",
]

# Mooncake FAST'25 traces (github.com/kvcache-ai/Mooncake) share the Bailian
# trace schema (timestamp/input_length/output_length/hash_ids), so the same
# block-hash reuse analysis applies; only the block size differs: Mooncake
# groups 512 tokens per hash id, not 16.
MOONCAKE_TOKENS_PER_BLOCK = 512
MOONCAKE_RAW_BASE = (
    "https://raw.githubusercontent.com/kvcache-ai/Mooncake/main/FAST25-release/traces"
)
MOONCAKE_TRACES = {
    "conversation": "conversation_trace.jsonl",
    "synthetic": "synthetic_trace.jsonl",
    "toolagent": "toolagent_trace.jsonl",
}


@dataclass
class BailianRecord:
    """One replayable request reconstructed from a Bailian/Qwen trace line.

    `reuse_prefix_blocks` is the count of leading blocks whose hash was already
    emitted by an earlier request, i.e. the contiguous prefix a KV cache could
    serve without recompute. It is computed in trace order before this request's
    own blocks are marked seen.
    """

    request_id: int
    chat_id: int
    parent_chat_id: int
    timestamp: float
    turn: int
    req_type: str
    input_length: int
    output_length: int
    hash_ids: list[int]
    prompt: str
    tokens: list[int]
    reuse_prefix_blocks: int
    is_reuse: bool


class TokenGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.special_ids = set(self.tokenizer.all_special_ids)
        # Precompute the non-special token-id pool ONCE. Rebuilding this 150k+
        # entry list per call made parsing O(unique_blocks * vocab_size); the
        # Bailian coder trace has ~5.2M unique blocks, which turned a parse into
        # a multi-hour CPU stall before any request ever reached the GPU.
        self.sample_ids = [v for k, v in self.vocab.items() if k not in self.special_ids]

    def get_random_tokens(self, length, decode=True):
        cur_ids = random.choices(self.sample_ids, k=length)
        # decode is the expensive part; callers that only need the ids (e.g. the
        # block reconstruction in parse_qwen_trace_records) skip it.
        cur_prompt = self.tokenizer.decode(cur_ids) if decode else None

        return cur_ids, cur_prompt


def parse_qwen_trace_records(trace_file, tokenizer, max_records=0, offset=0):
    """Reconstruct each Bailian/Qwen trace line into a replayable BailianRecord.

    Tokens are synthesized deterministically per block hash: equal hash ids across
    requests map to the *same* token block, so prefix reuse in the trace becomes
    real prefix reuse in the served prompts (and thus in any KV cache). The block
    reconstruction is the same algorithm the upstream (Rust) trace replayer uses.

    ``offset`` skips the first N trace lines in the returned window; ``max_records``
    caps the window size (0 = no cap). Reuse classification still depends on every
    earlier request, so lines before ``offset`` are parsed to keep block/seen state
    correct, then sliced off; the returned window's reuse flags match the full
    trace.
    """
    tokens_per_block = QWEN_TOKENS_PER_BLOCK
    # Lines needed to materialize the window [offset, offset+max_records). Reuse
    # classification only depends on earlier requests, so reading up to this bound
    # is equivalent to reading the whole trace then slicing, but avoids parsing
    # the whole 100k+ line trace when only a window is wanted. 0 = read all.
    read_limit = (offset + max_records) if (max_records and max_records > 0) else 0
    with open(trace_file, "r") as f:
        if read_limit > 0:
            traces = []
            for line in f:
                traces.append(json.loads(line))
                if len(traces) >= read_limit:
                    break
        else:
            traces = [json.loads(line) for line in f]

    token_generator = TokenGenerator(tokenizer)

    hash2token = {}
    token_block_dedup = set()
    seen_blocks = set()
    records = []

    for line_idx, trace in enumerate(traces):
        cur_prompt_tokens = []
        input_length = trace["input_length"]
        hash_ids = trace["hash_ids"]

        for cur_hash_id in hash_ids[:-1]:
            if cur_hash_id not in hash2token:
                block_tokens, _ = token_generator.get_random_tokens(
                    tokens_per_block, decode=False
                )
                block_token_str = ",".join(str(t) for t in block_tokens)
                while block_token_str in token_block_dedup:
                    block_tokens, _ = token_generator.get_random_tokens(
                        tokens_per_block, decode=False
                    )
                    block_token_str = ",".join(str(t) for t in block_tokens)
                hash2token[cur_hash_id] = block_tokens
                token_block_dedup.add(block_token_str)
            cur_prompt_tokens.extend(hash2token[cur_hash_id])

        last_hash_id = hash_ids[-1]
        # The trailing block is whatever is left after the leading full blocks.
        # input_length % block_size is wrong for exact multiples (it drops a full
        # 16-token block); derive the remainder from the block count instead.
        last_block_length = input_length - (len(hash_ids) - 1) * tokens_per_block
        last_block_length = max(0, min(last_block_length, tokens_per_block))
        if (last_hash_id not in hash2token) and (last_block_length > 0):
            block_tokens, _ = token_generator.get_random_tokens(
                last_block_length, decode=False
            )
            block_token_str = ",".join(str(t) for t in block_tokens)
            while block_token_str in token_block_dedup:
                block_tokens, _ = token_generator.get_random_tokens(
                    last_block_length, decode=False
                )
                block_token_str = ",".join(str(t) for t in block_tokens)
            hash2token[last_hash_id] = block_tokens
            token_block_dedup.add(block_token_str)
        if last_hash_id in hash2token:
            cur_prompt_tokens.extend(hash2token[last_hash_id])

        # Contiguous leading blocks already produced by an earlier request: the
        # prefix a KV cache could serve without recompute. Counted before this
        # request's own blocks are marked seen.
        reuse_prefix_blocks = 0
        for h in hash_ids:
            if h in seen_blocks:
                reuse_prefix_blocks += 1
            else:
                break
        seen_blocks.update(hash_ids)

        records.append(
            BailianRecord(
                request_id=line_idx,
                chat_id=trace.get("chat_id", line_idx),
                parent_chat_id=trace.get("parent_chat_id", -1),
                timestamp=float(trace.get("timestamp", 0.0)),
                turn=int(trace.get("turn", 0)),
                req_type=trace.get("type", "text"),
                input_length=input_length,
                output_length=int(trace.get("output_length", 0)),
                hash_ids=hash_ids,
                prompt=tokenizer.decode(cur_prompt_tokens),
                tokens=cur_prompt_tokens,
                reuse_prefix_blocks=reuse_prefix_blocks,
                is_reuse=reuse_prefix_blocks > 0,
            )
        )

        if (line_idx + 1) % 1000 == 0:
            print(f"Parsed {line_idx + 1} lines out of {len(traces)} lines.")

    if offset or (max_records and max_records > 0):
        end = (offset + max_records) if (max_records and max_records > 0) else None
        records = records[offset:end]
    return records


LONGBENCH_V2_HF_NAME = "THUDM/LongBench-v2"

# Faithful to the perf-instruct (non-CoT) framing of the official 0shot prompt:
# context first (the long, reusable prefix), then the question + 4 choices.
_LONGBENCH_V2_TEMPLATE = (
    "Please read the following text and answer the question below.\n\n"
    "{context}\n\n"
    "What is the correct answer to this question: {question}\n"
    "Choices:\n"
    "(A) {choice_A}\n(B) {choice_B}\n(C) {choice_C}\n(D) {choice_D}\n\n"
    "The correct answer is:"
)


@dataclass
class LongBenchV2Record:
    """One replayable LongBench v2 question.

    `is_reuse` is set when an earlier record carried a byte-identical context, so
    the long document prefix is already in the KV cache. `reuse_source_id` points
    at that first request and `reuse_prefix_len` is the context's token length:
    the prefix a cache serves without recompute. Natural reuse fraction is data
    dependent (only documents that back more than one question reuse).
    """

    request_id: int
    entry_id: str
    domain: str
    sub_domain: str
    difficulty: str
    question: str
    gold_answer: str
    prompt: str
    context_token_len: int
    reuse_source_id: int | None
    reuse_prefix_len: int
    is_reuse: bool


def _truncate_middle(context, tokenizer, max_context_tokens):
    """Keep head + tail of the context, dropping the middle, to fit a token budget.

    Mirrors LongBench's own runner: the answer-bearing spans tend to sit near the
    ends. Returns (possibly-truncated text, its token length).
    """
    token_ids = tokenizer.encode(context)
    if max_context_tokens <= 0 or len(token_ids) <= max_context_tokens:
        return context, len(token_ids)
    half = max_context_tokens // 2
    kept = token_ids[:half] + token_ids[-half:]
    return tokenizer.decode(kept), len(kept)


def parse_longbench_v2_records(
    tokenizer,
    max_entries=0,
    domain=None,
    difficulty=None,
    max_context_tokens=0,
):
    """Load LongBench v2 and classify per-record context reuse in dataset order.

    A record reuses the prefix of the first record sharing its exact context, so
    a sharer always trails its source (run_benchmark requires the source to have
    been issued already). The measured reuse fraction is printed because it is
    data dependent and load-bearing for the cache test.

    `max_context_tokens` (>0) middle-truncates each context to fit the model
    window. Reuse is keyed on the *original* context, so equal originals truncate
    identically and still share a prefix.
    """
    from datasets import load_dataset

    data = load_dataset(LONGBENCH_V2_HF_NAME, split="train")
    return classify_longbench_records(
        data,
        tokenizer,
        max_entries=max_entries,
        domain=domain,
        difficulty=difficulty,
        max_context_tokens=max_context_tokens,
    )


def classify_longbench_records(
    items,
    tokenizer,
    max_entries=0,
    domain=None,
    difficulty=None,
    max_context_tokens=0,
):
    """Classify per-record context reuse over an iterable of LongBench v2 dicts.

    Split out from the HuggingFace load so the reuse logic is testable offline.
    """
    first_seen: dict[str, tuple[int, int]] = {}  # context hash -> (req_id, ctx_tok_len)
    records: list[LongBenchV2Record] = []
    req_id = 0
    for item in items:
        if domain is not None and item.get("domain") != domain:
            continue
        if difficulty is not None and item.get("difficulty") != difficulty:
            continue

        context = item["context"]
        ctx_key = hashlib.sha1(context.encode("utf-8")).hexdigest()
        if ctx_key in first_seen:
            source_id, ctx_tok_len = first_seen[ctx_key]
            reuse_source_id = source_id
            is_reuse = True
            context, _ = _truncate_middle(context, tokenizer, max_context_tokens)
        else:
            context, ctx_tok_len = _truncate_middle(context, tokenizer, max_context_tokens)
            first_seen[ctx_key] = (req_id, ctx_tok_len)
            reuse_source_id = None
            is_reuse = False

        prompt = _LONGBENCH_V2_TEMPLATE.format(
            context=context,
            question=item["question"],
            choice_A=item["choice_A"],
            choice_B=item["choice_B"],
            choice_C=item["choice_C"],
            choice_D=item["choice_D"],
        )

        records.append(
            LongBenchV2Record(
                request_id=req_id,
                entry_id=str(item.get("_id", req_id)),
                domain=item.get("domain", ""),
                sub_domain=item.get("sub_domain", ""),
                difficulty=item.get("difficulty", ""),
                question=item["question"],
                gold_answer=item.get("answer", ""),
                prompt=prompt,
                context_token_len=ctx_tok_len,
                reuse_source_id=reuse_source_id,
                reuse_prefix_len=ctx_tok_len if is_reuse else 0,
                is_reuse=is_reuse,
            )
        )
        req_id += 1
        if max_entries > 0 and req_id >= max_entries:
            break

    n_reuse = sum(1 for r in records if r.is_reuse)
    frac = (n_reuse / len(records)) if records else 0.0
    print(
        f"LongBench v2: {len(records)} questions, {len(first_seen)} distinct "
        f"contexts, {n_reuse} reuse ({frac:.1%}). "
        + ("Low reuse: a context-sharing cache test will show a muted delta."
           if frac < 0.05 else "")
    )
    return records


@dataclass
class SCBenchRecord:
    """One replayable turn of a simplified multi-turn SCBench session.

    Turn 0 sends ``context + Q0`` and is fresh. Each later turn appends the prior
    turn's golden answer and the next question to the whole previous prompt, so the
    previous prompt is a byte-identical prefix already in the KV cache:
    ``reuse_source_id`` points at the previous turn and ``reuse_prefix_len`` is its
    token length. Golden (dataset) answers are used, not model output, so prompts
    are deterministic and the run is offline.
    """

    request_id: int
    session_id: int
    turn: int
    config: str
    prompt: str
    prompt_token_len: int
    context_token_len: int
    reuse_source_id: int | None
    reuse_prefix_len: int
    is_reuse: bool


def _scbench_to_text(value):
    """Stringify an SCBench field that may be text, an int-token sequence
    (scbench_mf contexts), or a numpy array (options/answers in some configs)."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return " ".join(str(v) for v in value)
    except TypeError:
        return str(value)


def _scbench_context_value(row):
    """Shared-context value across config schemas (scbench_vt stores it as input)."""
    for col in ("context", "input"):
        if col in row and row[col] is not None:
            return row[col]
    return None


def _scbench_turn_question(turn):
    """The user text for one turn: its input, plus multiple-choice options when
    the config carries them (scbench_choice_eng)."""
    question = _scbench_to_text(turn.get("input"))
    options = turn.get("options")
    if options is not None and len(options) > 0:
        letters = "ABCDEFGH"
        opts = "\n".join(
            f"{letters[i]}. {_scbench_to_text(o)}" for i, o in enumerate(options)
        )
        question = f"{question}\n{opts}"
    return question


def parse_scbench_records(
    df,
    config,
    tokenizer,
    max_sessions=0,
    max_turns=0,
    max_context_tokens=0,
):
    """Flatten SCBench sessions into simplified multi-turn replay records.

    Each session's long ``context`` is middle-truncated to ``max_context_tokens``
    (0 = no cap) and becomes the shared prefix. Turn k's prompt is the whole turn
    k-1 prompt plus that turn's golden answer and the next question, so KV reuse is
    the prior prompt. The appended block starts with a newline so the kept prefix's
    block hashes stay stable at the join (the last partial block may still differ,
    which only affects one block, mirroring longbench_v2_replay's filler).

    ``max_sessions``/``max_turns`` cap sessions and turns-per-session (0 = all).
    Reuse classification rides on reuse_source_id/reuse_prefix_len so the resulting
    RequestResult.is_prefix_reuse is set without a second pass.
    """
    records: list[SCBenchRecord] = []
    req_id = 0
    n_sessions = 0
    for session_id, (_, session) in enumerate(df.iterrows()):
        context = _scbench_to_text(_scbench_context_value(session))
        context, ctx_tok_len = _truncate_middle(context, tokenizer, max_context_tokens)
        turns = list(session["multi_turns"])
        if max_turns > 0:
            turns = turns[:max_turns]
        if not turns:
            continue

        prev_req_id = None
        prev_prompt = None
        prev_tok_len = 0
        for turn_idx, turn in enumerate(turns):
            question = _scbench_turn_question(turn)
            if turn_idx == 0:
                prompt = f"{context}\n\n{question}"
                reuse_source_id = None
                reuse_prefix_len = 0
                is_reuse = False
            else:
                prev_answer = _scbench_to_text(turns[turn_idx - 1].get("answer"))
                prompt = f"{prev_prompt}\n{prev_answer}\n\n{question}"
                reuse_source_id = prev_req_id
                reuse_prefix_len = prev_tok_len
                is_reuse = True

            tok_len = len(tokenizer.encode(prompt))
            records.append(
                SCBenchRecord(
                    request_id=req_id,
                    session_id=session_id,
                    turn=turn_idx,
                    config=config,
                    prompt=prompt,
                    prompt_token_len=tok_len,
                    context_token_len=ctx_tok_len,
                    reuse_source_id=reuse_source_id,
                    reuse_prefix_len=reuse_prefix_len,
                    is_reuse=is_reuse,
                )
            )
            prev_req_id = req_id
            prev_prompt = prompt
            prev_tok_len = tok_len
            req_id += 1

        n_sessions += 1
        if max_sessions > 0 and n_sessions >= max_sessions:
            break

    n_reuse = sum(1 for r in records if r.is_reuse)
    frac = (n_reuse / len(records)) if records else 0.0
    print(
        f"SCBench {config}: {n_sessions} sessions, {len(records)} turns, "
        f"{n_reuse} reuse ({frac:.1%})."
    )
    return records
