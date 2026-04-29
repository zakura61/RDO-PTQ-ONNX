# Quantization Analysis Detail: Code Evidence Map

This note is a code-backed companion to [quantization_analysis.md](quantization_analysis.md). Its purpose is narrow: show the exact places in code where quantization is explicitly inserted, and the places where it is absent.

Unless otherwise noted, the local file links below point to `RDO-original/`, so the evidence is anchored to the untouched original repository rather than the modified working tree.

The three questions covered here are:

1. Where do standard residual blocks explicitly requantize after residual fusion?
2. Where is `AttentionBlock` left unwrapped, causing internal and output-level fusion ops to remain outside explicit PTQ activation boundaries?
3. Why is the very input to `g_s` not coming from `ActQuantizer`?

## 1. Baseline: Where Quantization Is Explicitly Inserted

### 1.1 Generic quantized layers only quantize their own outputs

The generic quantization wrapper is `QuantModule`.

Code:

- [RDO-original/task-oriented-PTQ/quantization/quant_layer.py:107](RDO-original/task-oriented-PTQ/quantization/quant_layer.py:107)
- [RDO-original/task-oriented-PTQ/quantization/quant_layer.py:123](RDO-original/task-oriented-PTQ/quantization/quant_layer.py:123)
- [RDO-original/task-oriented-PTQ/quantization/quant_layer.py:130](RDO-original/task-oriented-PTQ/quantization/quant_layer.py:130)
- [RDO-original/task-oriented-PTQ/quantization/quant_layer.py:132](RDO-original/task-oriented-PTQ/quantization/quant_layer.py:132)

Relevant behavior:

```python
out = self.fwd_func(...)
out = self.activation_function(out)
if self.disable_act_quant:
    return out
if self.use_act_quant and self.trained:
    out = self.act_quantizer(out, True)
return out
```

Interpretation:

- `QuantModule` quantizes the output of the wrapped layer.
- It does not quantize arbitrary later tensor ops such as `out += identity` or `a * sigmoid(b)`.
- Therefore, if a higher-level block performs residual adds or muls outside a dedicated wrapper, those ops are outside the explicit activation-quant insertion points.

### 1.2 ResidualBlock has a dedicated quantized wrapper

Standard Cheng residual blocks are treated specially through `QuantRB`.

Code:

- [RDO-original/task-oriented-PTQ/quantization/quant_block.py:284](RDO-original/task-oriented-PTQ/quantization/quant_block.py:284)
- [RDO-original/task-oriented-PTQ/quantization/quant_block.py:301](RDO-original/task-oriented-PTQ/quantization/quant_block.py:301)
- [RDO-original/task-oriented-PTQ/quantization/quant_block.py:310](RDO-original/task-oriented-PTQ/quantization/quant_block.py:310)
- [RDO-original/task-oriented-PTQ/quantization/quant_block.py:312](RDO-original/task-oriented-PTQ/quantization/quant_block.py:312)

Relevant behavior:

```python
out = out + identity
if self.use_act_quant and self.trained:
    out = ActQuantizer(out)
return out
```

Interpretation:

- This is the positive control.
- For `ResidualBlock`, the repo explicitly inserts a quantizer after the residual add.
- So the residual fusion is inside an explicit PTQ boundary here.

## 2. Why `AttentionBlock` Is Different

### 2.1 `AttentionBlock` is absent from the `specials` registry

The quantization refactor only installs special block wrappers for types listed in `specials`.

Code:

- [RDO-original/task-oriented-PTQ/quantization/quant_model.py:37](RDO-original/task-oriented-PTQ/quantization/quant_model.py:37)
- [RDO-original/task-oriented-PTQ/quantization/quant_model.py:45](RDO-original/task-oriented-PTQ/quantization/quant_model.py:45)
- [RDO-original/task-oriented-PTQ/quantization/quant_block.py:648](RDO-original/task-oriented-PTQ/quantization/quant_block.py:648)

Relevant behavior:

```python
if type(child_module) in specials:
    setattr(module, name, specials[type(child_module)](...))
elif isinstance(child_module, (nn.Conv2d, ...)):
    setattr(module, name, QuantModule(...))
```

And the registry is:

```python
specials = {
    RSTB: QuantRSTB,
    ResidualBlockWithStride: QuantRBWS,
    ResidualBlockUpsample: QuantRBU,
    ResidualBlock: QuantRB,
    subpel_conv3x3: QuantSC,
}
```

Interpretation:

- `ResidualBlock` is special-cased.
- `AttentionBlock` is not.
- So `AttentionBlock` gets only recursive child replacement, not a dedicated wrapper with explicit post-fusion `ActQuantizer` calls.

## 3. Internal Attention Leak: Where Quantization Is Absent

This section refers to the official upstream `CompressAI` `AttentionBlock` implementation used by `compressai==1.2.4`, which this repo targets in `requirements.txt`.

Upstream source:

- `AttentionBlock` and `ResidualUnit`:
  https://interdigitalinc.github.io/CompressAI/_modules/compressai/layers/layers.html

What to inspect in upstream code:

- `ResidualUnit.forward()`
- `AttentionBlock.forward()`

What matters structurally:

- each internal `ResidualUnit` performs convolutional work, then residual fusion, then nonlinearity
- there is no repo-local `QuantAttentionBlock` wrapper that inserts `ActQuantizer` after those residual adds

Why this is visible from local code even without editing upstream:

- local quantization only wraps children recursively when a block type is not in `specials`
- child convs become `QuantModule`
- the parent block’s own tensor algebra remains untouched

Therefore:

- internal `ResidualUnit` conv outputs are quantized at their wrapped child outputs
- but the internal residual add itself is not followed by an explicit PTQ `ActQuantizer` insertion from this repo

## 4. Attention Output Leak: Where the Final Block Fusion Is Outside PTQ Boundary

Again, inspect upstream `AttentionBlock.forward()`:

- https://interdigitalinc.github.io/CompressAI/_modules/compressai/layers/layers.html

The important structure is:

- one branch produces `a`
- one branch produces `b`
- the block mixes them with a multiplicative gate
- the block then adds back the identity

Why this matters locally:

- there is no local wrapper analogous to `QuantRB` for `AttentionBlock`
- so there is no local code path that does:

```python
out = out + identity
out = ActQuantizer(out)
```

after the final attention-block fusion.

Code evidence for absence:

- [RDO-original/task-oriented-PTQ/quantization/quant_block.py:648](RDO-original/task-oriented-PTQ/quantization/quant_block.py:648)
- there is no `AttentionBlock: QuantSomething` entry in `specials`

Interpretation:

- the attention block output boundary is not explicitly requantized by this PTQ framework
- this is a real structural difference from `ResidualBlock`

## 5. `g_s` Very Input: Why It Does Not Come From `ActQuantizer`

### 5.1 `g_s` receives `y_hat`

Code:

- [RDO-original/task-oriented-PTQ/models/nic_cvt.py:300](RDO-original/task-oriented-PTQ/models/nic_cvt.py:300)
- [RDO-original/task-oriented-PTQ/models/nic_cvt.py:309](RDO-original/task-oriented-PTQ/models/nic_cvt.py:309)
- [RDO-original/task-oriented-PTQ/quantization/quant_block.py:202](RDO-original/task-oriented-PTQ/quantization/quant_block.py:202)
- [RDO-original/task-oriented-PTQ/quantization/quant_block.py:211](RDO-original/task-oriented-PTQ/quantization/quant_block.py:211)

Relevant behavior:

```python
y_hat = self.gaussian_conditional.quantize(
    y, "noise" if self.training else "dequantize"
)
x_hat = self.g_s(y_hat, x_size)
```

Interpretation:

- `g_s` does not receive the output of `ActQuantizer`.
- It receives `y_hat`, which comes from entropy-model quantize/dequantize.

### 5.2 In eval forward, `dequantize` returns rounded float values

Code:

- [RDO-original/task-oriented-PTQ/quantization/quantizer.py:19](RDO-original/task-oriented-PTQ/quantization/quantizer.py:19)
- [RDO-original/task-oriented-PTQ/quantization/quantizer.py:39](RDO-original/task-oriented-PTQ/quantization/quantizer.py:39)
- [RDO-original/task-oriented-PTQ/quantization/quantizer.py:41](RDO-original/task-oriented-PTQ/quantization/quantizer.py:41)

Relevant behavior:

```python
outputs = torch.round(outputs)
if mode == "dequantize":
    if means is not None:
        outputs += means
    return outputs
```

Interpretation:

- in the simple `forward()` path, `y_hat` is float dtype
- if no means are supplied to `quantize(..., "dequantize")`, it is a rounded float tensor
- this still is not an `ActQuantizer` boundary from the PTQ path

### 5.3 In real decode, `g_s` input can be non-integer float because of `means_hat`

Code:

- [RDO-original/task-oriented-PTQ/models/nic_cvt.py:558](RDO-original/task-oriented-PTQ/models/nic_cvt.py:558)
- [RDO-original/task-oriented-PTQ/models/nic_cvt.py:569](RDO-original/task-oriented-PTQ/models/nic_cvt.py:569)

Relevant behavior:

```python
rv = self.gaussian_conditional._dequantize(rv, means_hat)
...
x_hat = self.g_s(y_hat).clamp_(0, 1)
```

Interpretation:

- in true decompress, the tensor entering `g_s` is dequantized with `means_hat`
- that means the decoder entry tensor is float-domain and can be non-integer-valued
- this is why the `g_s` issue is best described as an entropy-dequantization entry-point mismatch, not as the exact same kind of missing post-add quantizer seen inside attention

## 6. Why `onnx_export.py` Shows Attention Gaps But Not `g_s` Entry Gaps

Code:

- [task-oriented-PTQ/onnx_export.py:21](task-oriented-PTQ/onnx_export.py:21)
- [task-oriented-PTQ/onnx_export.py:30](task-oriented-PTQ/onnx_export.py:30)
- [task-oriented-PTQ/onnx_export.py:48](task-oriented-PTQ/onnx_export.py:48)

Relevant behavior:

```python
model = Cheng2020Attention()
ga = q_model.model.g_a
...
elif target_name == 'attention':
    for m in ga:
        if isinstance(m, AttentionBlock):
            target_module = m
            break
```

Interpretation:

- the export script inspects encoder `g_a`
- for `attention`, it exports the first encoder `AttentionBlock`
- it does not inspect the entropy path that forms `y_hat`
- it does not inspect the decoder entry into `g_s`

So ONNX export is suitable for showing:

- missing `ActQuantizer` after internal attention add/mul boundaries

But not suitable for directly showing:

- that `g_s` starts from entropy dequantization rather than PTQ activation quantization

## 7. Bottom-Line Classification

### True: internal attention leak

Evidence:

- `AttentionBlock` absent from `specials`
- only child layers are quantized
- internal residual adds belong to the parent attention structure, not to `QuantRB`

### True: attention output leak

Evidence:

- no dedicated `AttentionBlock` quant wrapper exists
- therefore no explicit post-final-fusion `ActQuantizer` is inserted

### True: `g_s` very input mismatch

Evidence:

- `g_s` input is `y_hat`
- `y_hat` is produced by entropy quantize/dequantize, not `ActQuantizer`
- in decompress, it is dequantized with `means_hat`, so it can be non-integer float

## 8. Precision of Wording

To avoid overclaiming, the most accurate wording is:

- the code shows **missing explicit PTQ activation-quant boundaries**
- it does **not** prove that tensors are materialized as literal `int8` values throughout PyTorch execution
- in this repo, many quant boundaries are represented by fake-quant or atomic quantizer operators while tensors remain float dtype
