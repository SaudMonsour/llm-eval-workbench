# Rotary Attention Kit

NumPy rotary positional embeddings and causal attention with incremental-decoding equivalence checks.

---

## Overview

* **Task:** Verify a reference implementation of rotary positions and causal scaled dot-product attention.
* **Input:** 4 heads, 64 tokens, 32 values per head; seeded synthetic Gaussian tensors.
* **Scope:** Architecture unit study. No model training, language-quality score, or NVIDIA runtime benchmark.
* **NCA-GENL connection:** Transformer foundations, implementation testing, and deployment correctness.

---

## Key Results

| Check | Maximum absolute error |
| :--- | ---: |
| Full sequence vs incremental attention | 8.882e-16 |
| Shared position shift by 137 | 4.108e-15 |
| Vector norm after rotation | 8.882e-16 |
| Attention assigned to future keys | 0.0 |

The incremental path rotates keys using their original absolute positions and keeps all previous keys and values. Re-indexing cached keys from zero changes the computation. The tiny nonzero errors above are floating-point roundoff; causally masked probabilities are exactly zero.

## Design

For each adjacent pair, apply a 2D rotation by position times an inverse frequency. Query and key rotations share the same frequencies. Their dot product depends on the position difference. The implementation uses the adjacent-pair convention; checkpoints using a split-half layout need a conversion.

`rope(x, positions)` accepts `[heads, tokens, even_head_dim]`. `attention(q, k, v, q_positions, k_positions)` returns both outputs and probabilities. Explicit positions let a single new query attend to an existing prefix. Inputs and visible-key counts are validated.

## Visualizations

| Causal attention | Relative-position score |
| :---: | :---: |
| ![Causal probabilities](figures/attention.png) | ![Position dependent score](figures/relative-position.png) |

The score curve uses one fixed random query/key pair. It need not decrease monotonically with distance and does not demonstrate language understanding or context extrapolation.

## Repository Structure

* `attention.py`: reusable rotations and attention.
* `study.py`: executed numerical checks and charts.
* `test_attention.py`: six correctness and rejection tests.
* `metrics.json`, `audit.json`: measurements and execution evidence.
* `figures/`: two measured diagnostic plots.

## Limits and Sources

This CPU reference allocates the complete score matrix. It is not FlashAttention, a paged-cache allocator, or a complete transformer. Shift equivalence does not establish long-context generalization. Cross-validation and classification metrics do not apply to this numerical verification.

Sources: [RoFormer paper](https://arxiv.org/abs/2104.09864), [NVIDIA KV-cache documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/transformers-kv-cache.html), [NCA-GENL topics](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/).
