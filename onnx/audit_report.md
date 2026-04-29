# Quantization Audit Report: Cheng 2020 (RDO-PTQ)

## Executive Summary

This document records the findings from a structural and precision audit of the Cheng 2020 model quantization within the RDO-PTQ framework. The audit utilized "Atomic ONNX Export" to visualize the precision topology of the model.

## Critical Finding: Precision Inconsistency in Attention Mechanism

During the audit of the `AttentionBlock` (located at index 8 of the encoder `g_a`), a significant inconsistency was identified in how activation quantization is applied compared to standard residual blocks.

### 1. The Discrepancy

- **Standard Residual Blocks (`QuantRB`)**: These are top-level blocks explicitly wrapped by the framework. They include an `ActQuantizer` node after the residual addition (`out = out + identity`), maintaining strict 8-bit precision.
- **Attention Blocks (`AttentionBlock`)**: These are treated as generic containers. The framework recurses into them to quantize internal convolutions but fails to wrap the higher-level "Residual Units" and the final block output.

### 2. The "Nested Leak" Chain

The leak is recursive and occurs at multiple levels within the Attention mechanism:

#### Level A: Internal `ResidualUnit` Leaks

The `AttentionBlock` contains six `ResidualUnit` instances (3 in `conv_a`, 3 in `conv_b`). Each unit follows this logic:

1. `out = self.conv(x)` -> **Quantized (8-bit)**
2. `out += identity` -> **FP32 Addition (LEAK)**
3. `out = self.relu(out)` -> **FP32 ReLU (LEAK)**

#### Level B: Terminal Block Leak

After the internal units finish, the main `AttentionBlock` performs:

1. `out = a * torch.sigmoid(b)` -> **FP32 Multiplication (LEAK)**
2. `out += identity` -> **FP32 Addition (LEAK)**

### 3. Impact on Downstream Layers

Because of these nested leaks, the input to the **next** block in the encoder chain (the layer following the Attention mechanism) is not a quantized 8-bit tensor. It is a high-precision floating-point tensor that has accumulated precision from 8 separate unquantized additions/multiplications.

### 3. Potential Rationale vs. Oversight

- **Oversight Hypothesis**: The authors implemented specialized wrappers (`specials`) for `ResidualBlock`, `GDN`, and `subpel_conv3x3`, but likely omitted `AttentionBlock` from the `specials` registry in `quant_block.py`.
- **Impact**: This creates a non-uniform precision landscape where some residual sums are 8-bit and others are FP32, which may lead to unexpected behavior during low-bit inference or hardware deployment.

---

## Verification Guide: Auditing the Audit

To verify these findings, follow this roadmap comparing the refactored code against the original repository.

### 1. Reference Location

The original repository is cloned locally at:
`[workspace_root]/RDO-original/`

### 2. Manual Code Inspection

Compare the following files to see the discrepancy in wrapper logic:

- **Residual Blocks (Verified 8-bit)**:
  - **Refactored**: `task-oriented-PTQ/quantization/quant_block.py` (Line ~284, `QuantRB`)
  - **Original**: `RDO-original/task-oriented-PTQ/quantization/quant_block.py`
  - _Observation_: Both include `out = ActQuantizer(out)` after the sum.
- **Attention Blocks (FP32 Leak)**:
  - **Refactored**: `task-oriented-PTQ/quantization/quant_block.py` (Search for `specials` dictionary at the bottom)
  - **Original**: `RDO-original/task-oriented-PTQ/quantization/quant_block.py`
  - _Observation_: Notice that `AttentionBlock` is **missing** from the `specials` registry in both versions. This prevents the block from ever being wrapped in a quantized version of itself.

### 3. Visual Audit (Netron)

Generate the atomic graphs to see the leaks visually:

1. **Export the Attention Block**:

   ```bash
   python task-oriented-PTQ/onnx_export.py --target attention
   ```

2. **Inspection**: Open `cheng2020_attention_atomic.onnx` in Netron.
3. **What to look for**:
   - Follow the graph to the end of the block.
   - Observe that the final `Add` node (the residual sum) is **not** followed by an `ActQuantizer` node.
   - Contrast this with `cheng2020_convblock_atomic.onnx`, where every `Add` is followed by a clean `ActQuantizer` block.

---

## Technical Audit Artifacts

- **Isolated Export**: `cheng2020_attention_atomic.onnx`
- **Full Encoder Export**: `cheng2020_ga_atomic.onnx`
- **Reference Code**: `quantization/quant_block.py` vs `compressai/layers/layers.py`

## Auditor's Note

This finding was confirmed by comparing the refactored RDO-PTQ codebase against the original repository (`RDO-original`). The behavior is native to the original framework's design. No corrective action has been taken in the current workspace, as the objective is pure inspection.

## Addendum (2026-04-29): Clarified Leak Taxonomy and `g_s` Input Path

This section is appended after a second pass through the code to preserve the original audit text while recording a more precise understanding of where the precision boundaries are missing.

### A. Internal Attention Leak (`ResidualUnit` inside `AttentionBlock`)

This claim remains **true**.

Reason:

- `AttentionBlock` is not listed in the `specials` registry in `task-oriented-PTQ/quantization/quant_block.py`.
- Therefore, the quantization refactor only recurses into its child layers and replaces supported children such as convolutions with `QuantModule`.
- The higher-level residual add inside each internal `ResidualUnit` is not followed by a dedicated `ActQuantizer` insertion the way `QuantRB` does for `ResidualBlock`.

Practical meaning:

- The convolution outputs inside the internal attention substructure are quantized at their own module outputs.
- The residual fusion (`out += identity`) and the following nonlinearity occur outside an explicit PTQ activation-quant boundary.

### B. Attention Output Boundary Leak (`AttentionBlock` output itself)

This claim also remains **true**.

Reason:

- The outer `AttentionBlock` performs its own high-level mixing and residual fusion, but there is no `QuantAttentionBlock` wrapper in this repository.
- As a result, the final block output leaves the attention block without a dedicated `ActQuantizer` inserted after the final fusion.

This applies structurally to attention blocks in both encoder `g_a` and decoder `g_s`, not only to the specific encoder attention block exported during the first audit.

### C. `g_s` Entry-Point Precision Issue

This claim is **also true in substance**, but it should be described more carefully than "the input is int8" versus "the input is fp32".

Correct statement:

- The very input to `g_s` is `y_hat`.
- `y_hat` is not produced by the PTQ activation path (`ActQuantizer`).
- Instead, it is produced by the entropy-model quantize/dequantize path of `GaussianConditional`.

There are two relevant code paths:

1. `forward()` / evaluation path used by normal model execution:
   - `y_hat = self.gaussian_conditional.quantize(y, "dequantize")`
   - In this repository's helper implementation, `"dequantize"` returns `torch.round(y)` when no mean tensor is supplied.
   - Therefore, `y_hat` is a **float tensor with integer-valued entries**, not an activation-quantized tensor from the PTQ pipeline.

2. Real decode path (`decompress()`):
   - Symbols are decoded and then passed through `_dequantize(rv, means_hat)`.
   - This is effectively of the form `round(symbols) + means_hat`.
   - Therefore, the tensor fed into `g_s` is a **float tensor and can be non-integer-valued** because of the added `means_hat`.

Important terminology correction:

- The relevant offset here is `means_hat`, not `scales_hat`.
- So the most accurate mental model is closer to `round(y - means_hat) + means_hat` than `round(y - scale) + scale`.

Conclusion for `g_s`:

- The first layer of `g_s` receives a float-domain latent tensor coming from entropy dequantization.
- This is a real precision-boundary mismatch relative to the PTQ activation flow.
- It is a **different kind of issue** from the attention residual leak: this one is an entry-point mismatch at the start of the synthesis transform.

### Why `onnx_export.py` Did Not Show the `g_s` Input Issue

This is expected and does not invalidate the `g_s` observation.

Reason:

- The export script inspects `q_model.model.g_a` and selects submodules from the encoder path.
- For `--target attention`, it exports the first encoder `AttentionBlock` it encounters.
- It does **not** export the latent formation path that creates `y_hat`.
- It does **not** export the decoder entry path from entropy dequantization into `g_s`.

Therefore, the ONNX artifact is good at showing:

- missing `ActQuantizer` nodes after residual/add/mul operations inside the exported attention block

But it is not designed to show:

- how `y_hat` is formed before entering `g_s`
- whether the first decoder input comes from PTQ activation quantization or from entropy dequantization

### Updated Summary

At this point, the refined interpretation is:

- **Leak Type 1: Internal attention residual leak** -> **True**
- **Leak Type 2: Attention block output boundary leak** -> **True**
- **Leak Type 3: `g_s` very-input precision mismatch from entropy dequantization** -> **True**, but it should be described as a float-domain decoder entry-point issue rather than as the same kind of leak shown by the attention ONNX graph

### Scope Note

The wording "strict 8-bit tensor" should still be interpreted carefully throughout this document. In this codebase, many PTQ boundaries are represented by fake-quant / atomic quantizer operators while tensors remain in floating-point dtype during PyTorch execution and ONNX export.
