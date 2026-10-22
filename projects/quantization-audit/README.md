# Quantization Audit

Compare packed INT4 and INT8 weight storage, output drift, and holdout accuracy with a reusable NumPy quantizer.

---

## Overview

* **Task:** Weight-only post-training quantization audit.
* **Dataset:** 1,797 real handwritten digits, 64 pixel features, 10 classes.
* **Model:** Multinomial logistic regression trained on 1,437 samples; 360 held out.
* **NCA-GENL connection:** Model comparison, evaluation, compression, and deployment tradeoffs.

---

## Key Results

### Final Holdout Evaluation (360 Samples)

| Representation | Accuracy | Log loss | Parameter bytes | Prediction flips vs FP32 |
| :--- | ---: | ---: | ---: | ---: |
| FP32 | 0.9722 | 0.1020 | 2600 | 0 |
| INT8 tensor | 0.9722 | 0.1023 | 684 | 0 |
| INT8 channel | 0.9750 | 0.1035 | 720 | 1 |
| INT4 tensor | 0.9694 | 0.0919 | 364 | 6 |
| INT4 channel | 0.9556 | 0.1195 | 400 | 6 |

FP32 is the reference, not a dummy classifier. These are fixed-method comparisons, not a holdout-tuned model selection. INT8 tensor quantization preserved every reference prediction. INT4 channel quantization reduced logit RMSE relative to INT4 tensor quantization but had worse classification accuracy: output drift and accuracy measure different things. One extra correct prediction from INT8 channel is insufficient evidence of improved generalization.

## Data Summary

* **Source:** scikit-learn's bundled [digits dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html), originating from [UCI Optical Recognition of Handwritten Digits](https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits).
* **Preprocessing:** Stratified split with seed 42. Standardization is fitted only on training samples. The fitted scaler is included with the weights.
* **Comparison controls:** Identical trained weights, bias, holdout, and preprocessing for all representations. No seed search or calibration on the holdout.

## Design

Quantization rounds to signed symmetric ranges [-127,127] or [-7,7]. Per-channel scales refer to output columns. Two 4-bit signed values occupy each byte, including odd-length padding. Reported bytes include packed weights, float32 scales, and float32 biases. They exclude scaler arrays and file/container overhead, so they are not the size of a complete deployed application.

Inference unpacks and dequantizes weights before floating-point multiplication. Storage is compressed; CPU arithmetic is not an integer kernel. No latency or GPU speedup is claimed.

## Visualizations

| Holdout accuracy | Parameter storage |
| :---: | :---: |
| ![Accuracy](figures/accuracy.png) | ![Bytes](figures/parameter_bytes.png) |

## Repository Structure

* `quantize.py`: quantization, signed-nibble packing, and unpacking.
* `study.py`: real-data training, fixed comparisons, and charts.
* `test_quantize.py`: six tests covering zero weights, quantization error bounds, outliers, odd lengths, and invalid inputs.
* `weights.npz`: FP32 reference weights, bias, and training-fitted scaler.
* `metrics.json`, `audit.json`: results and execution evidence.

## Limits and Sources

This small classifier isolates quantization mechanics. It cannot establish LLM perplexity, factuality, or performance on NVIDIA GPUs. Cross-validation was not run; the design is a fixed post-training audit with one holdout, not a training model search. Tensor and channel scaling are different tradeoffs; neither is universally superior.

Sources: [NVIDIA quantization documentation](https://nvidia.github.io/TensorRT-LLM/latest/features/quantization.html), dataset links above, [NCA-GENL topics](https://www.nvidia.com/en-us/learn/certification/generative-ai-llm-associate/).
