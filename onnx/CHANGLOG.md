# Quantization Refactoring Log

This file documents changes made to the quantization framework to improve efficiency and ONNX exportability.

## [2026-04-29] Vectorized Activation Quantization

**File**: `task-oriented-PTQ/quantization/quantizer.py`

### Changes

- Refactored the `ActQuant` function to remove the channel-wise `for` loop.
- Implemented vectorized channel-wise quantization using `torch.amin` and `torch.amax`.
- Wrapped quantization logic in `torch.autograd.Function` with a custom `symbolic` method.
- Replaced in-place assignments with a clean functional flow.

### Rationale

- **Atomic ONNX Export**: The quantizer now appears as a single `ActQuantizer` node in ONNX, rather than a chain of arithmetic operations. This makes model auditing much easier.
- **Performance**: Vectorized operations are significantly faster on both CPU and GPU compared to Python loops.
