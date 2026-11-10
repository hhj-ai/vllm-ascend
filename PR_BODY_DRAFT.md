# 标题

[Model][Ops] Add InternLM2-20B-Chat support and dispatch dynamic NTK RoPE to the NPU kernel

# 正文

### What this PR does / why we need it?

Adds InternLM2-20B-Chat (`InternLM2ForCausalLM`, text-only dense GQA, 48 layers, BF16, 32K context) to vLLM Ascend, and fixes the rotary dispatch gap the model exposed.

**The rotary gap.** The checkpoint declares `rope_scaling = {"type": "dynamic", "factor": 2.0}`, so vLLM builds a `DynamicNTKScalingRotaryEmbedding`. `CustomOp.__new__` resolves the out-of-tree replacement **by class name**, and `REGISTERED_ASCEND_OPS` covered `RotaryEmbedding`, `MRotaryEmbedding`, `YaRNScalingRotaryEmbedding`, `DeepseekScalingRotaryEmbedding` and `Gemma4RotaryEmbedding` — but neither dynamic NTK variant. Every InternLM2 layer therefore ran `forward_native` on the NPU. Nothing in the log reports the fallback.

Measured on an Atlas A2 with the model's own config:

| rope configuration | class instantiated | NPU kernel |
|---|---|---|
| `dynamic` + `factor` (InternLM2) | `DynamicNTKScalingRotaryEmbedding` | no |
| `dynamic` + `alpha` (Hunyuan) | `DynamicNTKAlphaRotaryEmbedding` | no |
| default RoPE (control) | `AscendRotaryEmbedding` | yes |
| YaRN (control) | `AscendYaRNRotaryEmbedding` | yes |

This PR adds `AscendDynamicNTKScalingRotaryEmbedding` and `AscendDynamicNTKAlphaRotaryEmbedding` and registers both. Each subclasses its upstream class and overrides **only** the forward, so `_compute_cos_sin_cache` and `_compute_inv_freq` stay upstream's and the cos/sin table is unchanged — the NTK correction is applied once when the base is rescaled at init. The alpha variant covers Hunyuan checkpoints, which hit the same gap.

Both are pure wins, not trade-offs:

- Throughput, two cards, 1024 in / 128 out, 64 prompts, concurrency 8: **1.65 → 1.95 req/s (+18.7%)**, output **210.76 → 250.10 tok/s**, mean TPOT **34.30 → 28.89 ms (−15.8%)**, mean TTFT **483.5 → 408.5 ms**.
- Accuracy, bf16 against an fp32 reference, max absolute error: the fused kernel is closer than `forward_native` at every length tested (`1.29e-02` vs `1.85e-02` at 8 tokens, `2.52e-02` vs `2.95e-02` at 2048 tokens).

**The model.** `InternLM2ForCausalLM` is implemented upstream, so no model definition is added here. Beyond the rotary fix the model runs on the existing generic Ascend path.

Closes #10673
Related to #9079

### Does this PR introduce _any_ user-facing change?

Yes. InternLM2-20B-Chat is documented and covered by nightly accuracy tests. Separately, any checkpoint declaring a dynamic NTK `rope_scaling` now runs its rotary embedding on the NPU kernel.

A release note fragment is included.

### How was this patch tested?

**Unit tests** — `tests/ut/ops/test_rotary_embedding.py`, 63 passed. The new cases assert that constructing the upstream class yields the Ascend override (the regression this PR fixes), that neither cache hook is overridden, that the cache matches an independently written HF dynamic NTK formula bit for bit, and that `factor` with a defaulted `max_trained_positions` collapses to a plain RoPE cache.

**End-to-end, two Atlas A2 cards, TP=2, real weights** — 13/13 checks passed:

| check | result |
|---|---|
| `GET /v1/models` | 200 |
| `/v1/completions`, English and Chinese | 200, non-empty |
| greedy chat reproducible, English and Chinese | identical across repeats |
| ChatML markers absent from replies | no leak |
| sampling reproducible at a fixed seed | identical |
| sampling differs across seeds | differs |
| three-turn chat, fact recalled in turn 2 | recalled |
| third turn switching to Chinese | correct |
| retrieval, 73-token prompt | found |
| retrieval, 5749-token prompt | found |

`tests/e2e/pull_request/two_card/test_internlm2_20b_chat.py` carries the same coverage as repository tests: offline greedy determinism in both languages, seeded sampling behaviour, long-context retrieval, and an online multi-turn chat.

**Files changed**

- `vllm_ascend/ops/rotary_embedding.py`, `vllm_ascend/utils.py` — the two Ascend rotary subclasses and their registration.
- `tests/ut/ops/test_rotary_embedding.py` — unit tests.
- `tests/e2e/pull_request/two_card/test_internlm2_20b_chat.py` — serving tests.
- `tests/e2e/models/configs/InternLM2-20B-Chat.yaml`, `accuracy.txt` — C-Eval configuration and baseline.
- `.github/workflows/configs/nightly_config.yaml` — nightly two-card accuracy group.
- `.github/workflows/misc/model_dataset_list.json` — weight download entry, without which nightly cannot fetch the checkpoint.
- `docs/source/tutorials/models/InternLM2-20B-Chat.md`, `mkdocs.yml`, `supported_models.md` — tutorial and matrix entry.

<!-- 待补：C-Eval 实测值；CANN 9.2.1 环境下重跑的确认 -->
