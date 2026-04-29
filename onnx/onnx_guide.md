# Atomic ONNX Export Guide: RDO-PTQ

This guide explains the "Atomic Export" strategy used to transform complex quantization math into clean, interpretable nodes in Netron for auditing the Cheng 2020 model.

## The Problem: "Math Soup"

By default, the ONNX exporter decomposes quantization math (clamping, scaling, rounding) into dozens of primitive arithmetic nodes. This makes the model graph unreadable and impossible to audit for precision bottlenecks.

## The Solution: Atomic Custom Operators

We solved this using a three-layered approach to "hide" the math from the exporter.

### 1. Custom Operator Registration (`torch.library`)

We registered new operators in a custom namespace (`rdo`). By registering these as library operators, we tell PyTorch to treat them as single, atomic units during the graph tracing phase.

```python
# Defined in quantization/quantizer.py
torch.library.define("rdo::act_quant", "(Tensor x, int b_w) -> Tensor")

@torch.library.impl("rdo::act_quant", "default")
def act_quant_impl(x, b_w):
    return ActQuantFunction.apply(x, b_w) # Calls the real math
```

### 2. Symbolic Mapping (Legacy API)

We mapped these custom operators to high-level ONNX nodes. This is where we define the "name" that appears in Netron (e.g., `RDO::ActQuantizer`).

```python
def act_quant_symbolic(g, x, b_w):
    b_w_val = sym_help._get_const(b_w, 'i', 'b_w')
    # Maps to a single node in ONNX
    return g.op("RDO::ActQuantizer", x, bit_width_i=b_w_val).setType(x.type())

register_custom_op_symbolic("rdo::act_quant", act_quant_symbolic, 11)
```

### 3. The Export Strategy (Direct Legacy API)

In newer PyTorch versions (2.4+), the default `torch.onnx.export` uses a "Dynamo" engine that aggressively inlines custom functions. To bypass this and preserve our atomic nodes, we use the **Direct Legacy API**:

```python
from torch.onnx.utils import export

export(
    model,
    dummy_input,
    "model_atomic.onnx",
    opset_version=11,
    operator_export_type=torch.onnx.OperatorExportTypes.ONNX
)
```

## Summary of Atomic Nodes in Netron

| Atomic Node            | Python Function | Internal Math Hidden?         |
| :--------------------- | :-------------- | :---------------------------- |
| `RDO::ActQuantizer`    | `ActQuant()`    | Yes (Scale/Clamp/Round)       |
| `RDO::WeightQuantizer` | `WeightQuant()` | Yes (Channel-wise scaling)    |
| `RDO::GDN`             | `f_gdn()`       | Yes (Reparameterization math) |

## How to Audit specific blocks

Use the utility script:

```bash
python task-oriented-PTQ/onnx_export.py --target [ga|attention|convblock]
```

This isolates specific sub-modules to verify their local quantization topology.
