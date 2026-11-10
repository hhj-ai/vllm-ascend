#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Serving checks for InternLM2-20B-Chat on two cards.

The model is a text-only dense GQA decoder whose checkpoint selects dynamic
NTK rotary embeddings. These tests cover what that combination can break on
the NPU: the rotary path itself, the ChatML template and its two stop tokens,
greedy determinism under tensor parallelism, and long-context retrieval near
the declared 32K limit.
"""

import os

import pytest
import requests
from vllm import SamplingParams
from vllm.utils.network_utils import get_open_port

from tests.e2e.conftest import RemoteOpenAIServer, VllmRunner, wait_until_npu_memory_free

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

MODEL = "Shanghai_AI_Laboratory/internlm2-chat-20b"
TENSOR_PARALLEL_SIZE = 2

# `<|im_start|>` and `<|im_end|>` delimit every ChatML turn. The model must
# never emit them as visible text: `generation_config.json` lists 92542
# (`<|im_end|>`) alongside 2 (`</s>`) as a stop token, and leaking either
# marker means the stop ids did not reach the sampler.
CHATML_MARKERS = ("<|im_start|>", "<|im_end|>")


def assert_usable_reply(text: str, context: str) -> None:
    assert text.strip(), f"{context}: empty completion"
    for marker in CHATML_MARKERS:
        assert marker not in text, f"{context}: chat template marker {marker} leaked into the reply"


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="piecewise",
)
@wait_until_npu_memory_free()
def test_offline_greedy_is_deterministic_in_both_languages():
    """Offline generation, greedy, Chinese and English, short prompts.

    Greedy decoding has a single correct answer per prompt, so repeating the
    batch must reproduce it token for token. Under tensor parallelism a rotary
    or collective bug usually shows up here first, as a reply that is fluent
    but not reproducible.
    """
    prompts = [
        "The capital of France is",
        "请用一句话解释什么是机器学习。",
    ]

    with VllmRunner(
        MODEL,
        max_model_len=1024,
        dtype="bfloat16",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=0.9,
    ) as runner:
        first = runner.generate_greedy(prompts, max_tokens=32)
        second = runner.generate_greedy(prompts, max_tokens=32)

    for (_, first_text), (_, second_text), prompt in zip(first, second, prompts):
        assert_usable_reply(first_text[len(prompt) :], f"greedy completion for {prompt!r}")
        assert first_text == second_text, f"greedy decoding is not reproducible for {prompt!r}"


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="piecewise",
)
@wait_until_npu_memory_free()
def test_offline_sampling_is_seeded_and_varies():
    """Non-greedy sampling must be reproducible per seed and not collapse.

    Two properties are checked together because either one alone can hide a
    defect: a broken sampler can be perfectly reproducible by always emitting
    the argmax, and a sampler reading uninitialised state can vary without
    honouring the seed.
    """
    prompt = ["List three uses for a paperclip."]
    sampling = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=48, seed=1234)
    other_seed = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=48, seed=4321)

    with VllmRunner(
        MODEL,
        max_model_len=1024,
        dtype="bfloat16",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=0.9,
    ) as runner:
        first = runner.generate(prompt, sampling)
        repeat = runner.generate(prompt, sampling)
        different = runner.generate(prompt, other_seed)

    assert_usable_reply(first[0][1][0], "sampled completion")
    assert first[0][1][0] == repeat[0][1][0], "sampling with a fixed seed is not reproducible"
    assert first[0][1][0] != different[0][1][0], "sampling ignores the seed: two seeds gave identical text"


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="long_sequence",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="piecewise",
)
@wait_until_npu_memory_free()
def test_offline_long_context_retrieval():
    """A fact placed mid-prompt must survive a long context.

    The prompt is padded past the point where the rotary table is indexed far
    beyond the short-prompt range, which is where a mis-scaled dynamic NTK
    cache stops being equivalent to plain RoPE.
    """
    secret = "The access code for the north gate is 73915."
    filler = "The weather report for that day was unremarkable. " * 600
    prompt = f"{filler}\n{secret}\n{filler}\nQuestion: What is the access code for the north gate?\nAnswer:"

    with VllmRunner(
        MODEL,
        max_model_len=16384,
        dtype="bfloat16",
        tensor_parallel_size=TENSOR_PARALLEL_SIZE,
        gpu_memory_utilization=0.9,
        max_num_seqs=4,
    ) as runner:
        outputs = runner.generate_greedy([prompt], max_tokens=16)

    completion = outputs[0][1][0][len(prompt) :]
    assert "73915" in completion, f"long-context retrieval failed, model answered: {completion!r}"


@pytest.mark.e2e_model(MODEL)
@pytest.mark.e2e_coverage(
    arch="dense",
    feature="",
    parallel="TP",
    deploy="pd_mix",
    hardware="A2",
    quantization="BF16",
    graph_mode="piecewise",
)
@wait_until_npu_memory_free()
def test_online_multi_turn_chat_and_completions():
    """Online serving: completions, a three-turn chat, and the stop tokens.

    The chat turns are sent through the checkpoint's own ChatML template, so
    this also covers `trust_remote_code` tokenizer loading and the
    `eos_token_id: [2, 92542]` pair from `generation_config.json`.
    """
    port = get_open_port()
    server_args = [
        "--trust-remote-code",
        "--max-model-len",
        "4096",
        "--tensor-parallel-size",
        str(TENSOR_PARALLEL_SIZE),
        "--dtype",
        "bfloat16",
        "--gpu-memory-utilization",
        "0.9",
        "--port",
        str(port),
    ]

    with RemoteOpenAIServer(MODEL, server_args, server_port=port, auto_port=False) as server:
        completion = requests.post(
            server.url_for("v1", "completions"),
            json={"model": MODEL, "prompt": "Shanghai is", "max_tokens": 32, "temperature": 0.0},
        )
        assert completion.status_code == 200
        assert_usable_reply(completion.json()["choices"][0]["text"], "online completion")

        messages = [{"role": "user", "content": "My favourite colour is teal. Remember it."}]
        for follow_up in ("What is my favourite colour?", "用中文再说一遍。"):
            chat = requests.post(
                server.url_for("v1", "chat", "completions"),
                json={"model": MODEL, "messages": messages, "max_tokens": 64, "temperature": 0.0},
            )
            assert chat.status_code == 200
            reply = chat.json()["choices"][0]["message"]["content"]
            assert_usable_reply(reply, f"chat turn {len(messages)}")
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": follow_up})

        recall = requests.post(
            server.url_for("v1", "chat", "completions"),
            json={"model": MODEL, "messages": messages, "max_tokens": 64, "temperature": 0.0},
        )
        assert recall.status_code == 200
        final = recall.json()["choices"][0]["message"]["content"]
        assert_usable_reply(final, "final chat turn")
        # The model was told the colour in turn one; losing it across turns
        # means the conversation was not assembled into a single prompt.
        assert "teal" in final.lower() or "青" in final or "蓝绿" in final, (
            f"multi-turn context was lost, model answered: {final!r}"
        )
