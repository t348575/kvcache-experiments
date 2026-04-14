import copy
import random
import json
import os
import pickle

CACHE_DIR = "./cached_data"


class DummyLongDocDataset:
    def __init__(
        self,
        document_length,
        num_documents,
        document_repeat,
        document_repeat_mode,
        shuffle_seed=0,
        max_entries=None,
    ):
        self.document_length = document_length
        self.num_documents = num_documents
        self.document_repeat = document_repeat
        self.document_repeat_mode = document_repeat_mode
        self.shuffle_seed = shuffle_seed

        self.dataset_size = num_documents * document_repeat
        if max_entries is not None and max_entries < self.dataset_size:
            self.dataset_size = max_entries
        self.index = 0
        self.data = self._compose_dataset()

    def _compose_dataset(self):
        prompts = [
            str(i) + " ".join(["hi"] * self.document_length)
            for i in range(self.num_documents)
        ]

        if self.document_repeat_mode == "random":
            prompts = [p for p in prompts for _ in range(self.document_repeat)]
            random.Random(self.shuffle_seed).shuffle(prompts)
        elif self.document_repeat_mode == "min-distance":
            prompts = [p for p in prompts for _ in range(self.document_repeat)]
        elif self.document_repeat_mode == "max-distance":
            prompts = prompts * self.document_repeat
        else:
            raise ValueError(
                f"Unsupported document_repeat_mode: {self.document_repeat_mode}"
            )

        return prompts

    def size(self):
        return self.dataset_size

    def next_item(self):
        if self.index >= self.dataset_size:
            return None

        prompt = self.data[self.index]
        self.index += 1

        return prompt


class RandomizedPrefixDataset:
    def __init__(
        self,
        tokenizer,
        document_length,
        num_documents,
        document_repeat,
        document_repeat_mode,
        shuffle_seed=0,
        max_entries=None,
        prompts=None,
    ):
        self.tokenizer = tokenizer
        self.document_length = document_length
        self.num_documents = num_documents
        self.document_repeat = document_repeat
        self.document_repeat_mode = document_repeat_mode
        self.shuffle_seed = shuffle_seed

        if prompts is not None:
            # Build the dataset from an existing list of prompts
            self.num_documents = len(prompts)
            self.document_repeat = 1
            self.document_repeat_mode = "min-distance"

            self.dataset_size = len(prompts)
            if max_entries is not None and max_entries < self.dataset_size:
                self.dataset_size = max_entries
            self.index = 0
            self.data_dedup = copy.deepcopy(prompts)
            self.data = copy.deepcopy(prompts)
            return

        self.dataset_size = num_documents * document_repeat
        if max_entries is not None and max_entries < self.dataset_size:
            self.dataset_size = max_entries
        self.index = 0
        self.data_dedup, self.data = self._compose_dataset()

    def _compose_dataset(self):
        vocab = self.tokenizer.get_vocab()
        special_ids = set(self.tokenizer.all_special_ids)

        prompts = []
        for i in range(self.num_documents):
            cur_ids = random.choices(
                [v for k, v in vocab.items() if k not in special_ids],
                k=self.document_length,
            )
            cur_prompt = f"{i} " + self.tokenizer.decode(cur_ids)
            cur_ids = self.tokenizer.encode(cur_prompt)[: self.document_length]
            cur_prompt = self.tokenizer.decode(cur_ids)
            prompts.append(cur_prompt)

        prompts_dedup = copy.deepcopy(prompts)

        if self.document_repeat_mode == "random":
            prompts = [p for p in prompts for _ in range(self.document_repeat)]
            random.Random(self.shuffle_seed).shuffle(prompts)
        elif self.document_repeat_mode == "min-distance":
            prompts = [p for p in prompts for _ in range(self.document_repeat)]
        elif self.document_repeat_mode == "max-distance":
            prompts = prompts * self.document_repeat
        else:
            raise ValueError(
                f"Unsupported document_repeat_mode: {self.document_repeat_mode}"
            )

        return prompts_dedup, prompts

    def get_dedup_dataset(self):
        return RandomizedPrefixDataset(
            tokenizer=self.tokenizer,
            document_length=self.document_length,
            num_documents=len(self.data_dedup),
            document_repeat=1,
            document_repeat_mode="min-distance",
            max_entries=self.dataset_size,
            prompts=self.data_dedup,
        )

    def size(self):
        return self.dataset_size

    def next_item(self):
        if self.index >= self.dataset_size:
            return None

        prompt = self.data[self.index]
        self.index += 1

        return prompt


class LooGLEDataset:
    """
    https://github.com/bigai-nlco/LooGLE
    The map between the two version of names:
    * context <-> input
    * question <-> Q
    * answer <-> A
    * evidence <-> S
    """

    def __init__(self, task, max_entries=None):
        from datasets import load_dataset

        datasets = ["shortdep_qa", "shortdep_cloze", "longdep_qa", "summarization"]
        if task not in datasets:
            raise ValueError(f"Unsupported dataset class: {task}")

        self.data = load_dataset("bigainlco/LooGLE", task, split="test")
        self.task = task
        self.dataset_size = len(self.data)
        if max_entries is not None and max_entries < self.dataset_size:
            self.dataset_size = max_entries
        self.index = 0
        self.iter_index = 0

        if task in ["shortdep_qa", "longdep_qa"]:
            self.compose_prompt = self._compose_dep_qa
        elif task in ["summarization"]:
            self.compose_prompt = self._compose_summarization
        elif task in ["shortdep_cloze"]:
            self.compose_prompt = self._compose_shortdep_cloze

    @staticmethod
    def _compose_dep_qa(data_item):
        prompt = "Please answer the question based on the long texts below."
        prompt += "\n{}\nQuestion: {}\nAnswer: ".format(
            data_item["context"], data_item["question"]
        )
        return prompt

    @staticmethod
    def _compose_summarization(data_item):
        prompt = "Please generate a summary of the below paper. \n"
        prompt += "{}\n Summarization: ".format(data_item["context"])
        return prompt

    @staticmethod
    def _compose_shortdep_cloze(data_item):
        prompt = "Please fill in the clozes based on the given long texts below. Each of the placeholder '<mask-n>' in the question could be an entity of Person, Location or Organiocation. The same masks represent the same entity. Output a json format answer, for example: {{'<mask-0>': 'Bob', '<mask-1>': 'Gorrosion Magazine','<mask-2>': 'Bethel Horizon'}}\n"
        prompt += "{}\n Question: {} What are the masked entities? \nAnswer:".format(
            data_item["context"], data_item["question"]
        )

        return prompt

    def next_item(self):
        # TODO: return something that can be direclty used in by the model
        if self.index >= self.dataset_size:
            return None

        data_item = self.data[self.index]
        self.index += 1

        prompt = self.compose_prompt(data_item)

        return prompt

    def __iter__(self):
        while self.iter_index < self.dataset_size:
            data_item = self.data[self.iter_index]
            self.iter_index += 1

            prompt = self.compose_prompt(data_item)

            yield prompt


class TokenGenerator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.vocab = self.tokenizer.get_vocab()
        self.special_ids = set(self.tokenizer.all_special_ids)

    def get_random_tokens(self, length):

        cur_ids = random.choices(
            [v for k, v in self.vocab.items() if k not in self.special_ids],
            k=length,
        )
        cur_prompt = self.tokenizer.decode(cur_ids)

        return cur_ids, cur_prompt


def parse_mooncake_trace_file(trace_file, tokenizer, max_prompt_length=None):
    tokens_per_block = 512  # Now we do not allow this to be changed

    with open(trace_file, "r") as f:
        traces = [json.loads(line) for line in f]

    token_generator = TokenGenerator(tokenizer)

    # Assign deterministic but unique token sequences per hash_id
    hash2token = {}
    token_block_dedup = set()
    prompts = []
    tokens = []
    parsed_lines = 0

    # Build prompts
    for trace in traces:
        parsed_lines += 1
        cur_prompt_tokens = []
        cur_num_tokens = trace["input_length"]
        for cur_hash_id in trace["hash_ids"][:-1]:
            if not (cur_hash_id in hash2token):
                block_tokens, _ = token_generator.get_random_tokens(tokens_per_block)
                block_token_str = ",".join([str(t) for t in block_tokens])
                # although it is unlikely that we will have collisions, we still check
                while block_token_str in token_block_dedup:
                    print(
                        f"#debug {parsed_lines}: hash id collision detected, regenerating tokens"
                    )
                    block_tokens, _ = token_generator.get_random_tokens(
                        tokens_per_block
                    )
                    block_token_str = ",".join([str(t) for t in block_tokens])
                hash2token[cur_hash_id] = block_tokens
                token_block_dedup.add(block_token_str)
            cur_prompt_tokens.extend(hash2token[cur_hash_id])

        # last hashid
        last_hash_id = trace["hash_ids"][-1]
        last_block_length = cur_num_tokens % 512
        if (not (last_hash_id in hash2token)) and (last_block_length > 0):
            block_tokens, _ = token_generator.get_random_tokens(last_block_length)
            block_token_str = ",".join([str(t) for t in block_tokens])
            # although it is unlikely that we will have collisions, we still check
            while block_token_str in token_block_dedup:
                print(
                    f"#debug-{parsed_lines}-{last_block_length}: hash id collision detected, regenerating tokens"
                )
                block_tokens, _ = token_generator.get_random_tokens(last_block_length)
                block_token_str = ",".join([str(t) for t in block_tokens])

            hash2token[last_hash_id] = block_tokens
            token_block_dedup.add(block_token_str)
            cur_prompt_tokens.extend(hash2token[last_hash_id])

        # print(
        #     f"#debug: line {parsed_lines} has {len(cur_prompt_tokens)} tokens for prompt length {cur_num_tokens}"
        # )

        prompts.append(tokenizer.decode(cur_prompt_tokens))
        tokens.append(cur_prompt_tokens)

        test_num_tokens = tokenizer.encode(tokenizer.decode(cur_prompt_tokens))
        if test_num_tokens != cur_prompt_tokens:
            raise ValueError(
                f"Tokenization mismatch at line {parsed_lines}, expected {len(cur_prompt_tokens)} tokens, got {len(test_num_tokens)} tokens"
            )

        if parsed_lines % 1000 == 0:
            print(f"Parsed {parsed_lines} lines out of {len(traces)} lines.")

    return prompts, tokens


class MooncakeTraceDataset:

    def __init__(
        self,
        path,
        task,
        tokenizer,
        max_entries=0,
        max_prompt_length=None,
        cache_file_label="",
        overwrite_cache=False,
    ):
        tasks = ["conversation", "synthetic", "toolagent", "test"]
        if task not in tasks:
            raise ValueError(f"Unsupported dataset class: {task}")

        trace_file_path = os.path.join(path, f"{task}_trace.jsonl")
        cached_data_dir = os.path.join(CACHE_DIR, "mooncake_traces")
        os.makedirs(cached_data_dir, exist_ok=True)
        cached_data_path = os.path.join(
            cached_data_dir, f"{task}_{cache_file_label}_parsed_hash.pkl"
        )

        generate_tokens = False
        if os.path.exists(cached_data_path):
            if not overwrite_cache:
                with open(cached_data_path, "rb") as f:
                    self.prompts, self.tokens = pickle.load(f)
                print(f"Loaded cached Mooncake trace dataset from {cached_data_path}")
            else:
                os.remove(cached_data_path)
                generate_tokens = True
        else:
            generate_tokens = True

        if generate_tokens:
            print("Parsing Mooncake trace dataset...")
            self.prompts, self.tokens = parse_mooncake_trace_file(
                trace_file_path,
                tokenizer=tokenizer,
            )
            with open(cached_data_path, "wb") as f:
                pickle.dump((self.prompts, self.tokens), f)
            print(f"Cached Mooncake trace dataset to {cached_data_path}")

        # Truncate prompts if needed
        if max_prompt_length is not None:
            truncated_prompts = []
            truncated_tokens = []
            for prompt, token in zip(self.prompts, self.tokens):
                if len(token) <= max_prompt_length:
                    truncated_prompts.append(prompt)
                    truncated_tokens.append(token)
                else:
                    truncated_token = token[:max_prompt_length]
                    truncated_prompt = tokenizer.decode(truncated_token)
                    truncated_prompts.append(truncated_prompt)
                    truncated_tokens.append(truncated_token)
            self.prompts = truncated_prompts
            self.tokens = truncated_tokens

        self.task = task
        self.dataset_size = len(self.prompts)
        if max_entries > 0 and max_entries < self.dataset_size:
            self.dataset_size = max_entries
        self.max_prompt_length = max_prompt_length

        self.index = 0
        self.iter_index = 0

    def size(self):
        return self.dataset_size

    def next_item(self):
        if self.index >= self.dataset_size:
            return None

        prompt = self.prompts[self.index]
        self.index += 1

        return prompt

    def next_token_list(self):
        if self.index >= self.dataset_size:
            return None

        token_list = self.tokens[self.index]
        self.index += 1

        return token_list

    def __iter__(self):
        while self.iter_index < self.dataset_size:
            prompt = self.prompts[self.iter_index]
            self.iter_index += 1

            yield prompt


def parse_qwen_trace_file(trace_file, tokenizer):
    tokens_per_block = 16
    with open(trace_file, "r") as f:
        traces = [json.loads(line) for line in f]

    token_generator = TokenGenerator(tokenizer)

    # Assign deterministic but unique token sequences per hash_id
    hash2token = {}
    token_block_dedup = set()
    prompts = []
    tokens = []
    parsed_lines = 0

    # Build prompts
    for trace in traces:
        parsed_lines += 1
        cur_prompt_tokens = []
        input_length = trace["input_length"]
        for cur_hash_id in trace["hash_ids"][:-1]:
            if not (cur_hash_id in hash2token):
                block_tokens, _ = token_generator.get_random_tokens(tokens_per_block)
                block_token_str = ",".join([str(t) for t in block_tokens])
                # although it is unlikely that we will have collisions, we still check
                while block_token_str in token_block_dedup:
                    print(
                        f"#debug {parsed_lines}: hash id collision detected, regenerating tokens"
                    )
                    block_tokens, _ = token_generator.get_random_tokens(
                        tokens_per_block
                    )
                    block_token_str = ",".join([str(t) for t in block_tokens])
                hash2token[cur_hash_id] = block_tokens
                token_block_dedup.add(block_token_str)
            cur_prompt_tokens.extend(hash2token[cur_hash_id])

        last_hash_id = trace["hash_ids"][-1]
        last_block_length = input_length % tokens_per_block
        if (not (last_hash_id in hash2token)) and (last_block_length > 0):
            block_tokens, _ = token_generator.get_random_tokens(last_block_length)
            block_token_str = ",".join([str(t) for t in block_tokens])
            # although it is unlikely that we will have collisions, we still check
            while block_token_str in token_block_dedup:
                print(
                    f"#debug-{parsed_lines}-{last_block_length}: hash id collision detected, regenerating tokens"
                )
                block_tokens, _ = token_generator.get_random_tokens(last_block_length)
                block_token_str = ",".join([str(t) for t in block_tokens])

            hash2token[last_hash_id] = block_tokens
            token_block_dedup.add(block_token_str)
            cur_prompt_tokens.extend(hash2token[last_hash_id])

        prompts.append(tokenizer.decode(cur_prompt_tokens))
        tokens.append(cur_prompt_tokens)

        if parsed_lines % 1000 == 0:
            print(f"Parsed {parsed_lines} lines out of {len(traces)} lines.")

    return prompts, tokens


class QwenTraceDataset:
    def __init__(
        self,
        path,
        task,
        tokenizer,
        max_entries=0,
        overwrite_cache=False,
        cache_file_label="",
        max_prompt_length=None,
    ):
        tasks = ["A", "B", "test"]
        if task not in tasks:
            raise ValueError(f"Unsupported dataset class: {task}")

        trace_file_path = os.path.join(path, f"qwen_trace{task}_blksz_16.jsonl")
        cached_data_dir = os.path.join(CACHE_DIR, "qwen_traces")
        os.makedirs(cached_data_dir, exist_ok=True)
        cached_data_path = os.path.join(
            cached_data_dir, f"{task}_{cache_file_label}_parsed_hash_unit_{16}.pkl"
        )

        if os.path.exists(cached_data_path):
            if not overwrite_cache:
                with open(cached_data_path, "rb") as f:
                    self.prompts, self.tokens = pickle.load(f)
                print(f"Loaded cached Qwen trace dataset from {cached_data_path}")
            else:
                os.remove(cached_data_path)
        else:
            self.prompts, self.tokens = parse_qwen_trace_file(
                trace_file_path, tokenizer=tokenizer
            )
            with open(cached_data_path, "wb") as f:
                pickle.dump((self.prompts, self.tokens), f)
            print(f"Cached Qwen trace dataset to {cached_data_path}")

        if max_prompt_length is not None:
            truncated_prompts = []
            truncated_tokens = []
            for prompt, token in zip(self.prompts, self.tokens):
                if len(token) <= max_prompt_length:
                    truncated_prompts.append(prompt)
                    truncated_tokens.append(token)
                else:
                    truncated_token = token[:max_prompt_length]
                    truncated_prompt = tokenizer.decode(truncated_token)
                    truncated_prompts.append(truncated_prompt)
                    truncated_tokens.append(truncated_token)
            self.prompts = truncated_prompts
            self.tokens = truncated_tokens

        self.task = task
        self.dataset_size = len(self.prompts)
        if max_entries > 0 and max_entries < self.dataset_size:
            self.dataset_size = max_entries
        self.max_prompt_length = max_prompt_length

        self.index = 0
        self.iter_index = 0

    def size(self):
        return self.dataset_size

    def next_item(self):
        if self.index >= self.dataset_size:
            return None

        prompt = self.prompts[self.index]
        tokens = self.tokens[self.index]
        self.index += 1

        while len(tokens) > self.max_length:
            if self.index >= self.dataset_size:
                return None
            prompt = self.prompts[self.index]
            tokens = self.tokens[self.index]
            self.index += 1

        return prompt

    def __iter__(self):
        while self.iter_index < self.dataset_size:
            prompt = self.prompts[self.iter_index]
            self.iter_index += 1

            yield prompt


if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

    prompts = parse_mooncake_trace_file("./trace_sample.txt", tokenizer=tokenizer)

    for l in prompts:
        print(l)
        print("-----")
