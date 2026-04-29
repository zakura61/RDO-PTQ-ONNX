import torch
import torch.nn as nn
import sys
import os
import argparse

# Add paths to sys.path to allow imports
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
target_dir = os.path.join(project_root, 'task-oriented-PTQ')
if target_dir not in sys.path:
    sys.path.append(target_dir)

from compressai.models import Cheng2020Attention
from quantization.quant_model import QuantModel
from quantization.quant_block import QuantRB, QuantRBWS
from compressai.layers.layers import AttentionBlock

def export_target(target_name):
    print(f"Targeting: {target_name}")
    
    # 1. Setup model
    model = Cheng2020Attention()
    q_model = QuantModel(model, weight_quant_params={'n_bits': 8}, act_quant_params={'n_bits': 8})
    q_model.set_quant_state(True, True)
    
    # Ensure 'trained' mode is on for activation quantization logic to trigger
    for m in q_model.modules():
        if hasattr(m, 'trained'):
            m.trained = True
    
    # 2. Select the specific submodule
    ga = q_model.model.g_a
    
    target_module = None
    file_name = f"cheng2020_{target_name}_atomic.onnx"

    if target_name == 'ga':
        target_module = ga
    elif target_name == 'convblock':
        for m in ga:
            if isinstance(m, QuantRB):
                target_module = m
                break
    elif target_name == 'convupblock':
        for m in ga:
            if isinstance(m, QuantRBWS):
                target_module = m
                break
    elif target_name == 'attention':
        for m in ga:
            if isinstance(m, AttentionBlock):
                target_module = m
                break
    
    if target_module is None:
        print(f"Error: Could not find target '{target_name}' in the model.")
        return

    # 3. Auto-detect input channels for the target module
    in_channels = 3 # Default fallback
    found_channels = False
    
    # Iterate through submodules to find the first one with weights
    for m in target_module.modules():
        # Check for regular Conv or our QuantModule
        if hasattr(m, 'weight') and m.weight is not None and m.weight.ndim == 4:
            in_channels = m.weight.size(1)
            found_channels = True
            break
            
    # Special case: If we are targeting the full ga, it MUST be 3 channels
    if target_name == 'ga':
        in_channels = 3
        dummy_input = torch.randn(1, 3, 256, 256)
    elif target_name == 'attention':
        dummy_input = torch.randn(1, in_channels, 16, 16)
    else:
        dummy_input = torch.randn(1, in_channels, 64, 64)

    # 4. Warm up to initialize parameters
    print(f"Initializing {target_name} parameters (Input channels: {in_channels})...")
    target_module.eval()
    with torch.no_grad():
        _ = target_module(dummy_input)
    
    # 5. Export using Direct Legacy API to force atomic symbolic nodes
    print(f"Exporting {target_name} to ATOMIC ONNX...")
    try:
        from torch.onnx.utils import export
        export(
            target_module,
            dummy_input,
            file_name,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            operator_export_type=torch.onnx.OperatorExportTypes.ONNX
        )
        print(f"Export successful: {file_name}")
    except Exception as e:
        print(f"Export failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Atomic ONNX Exporter for Cheng 2020 Audit")
    parser.add_argument("--target", type=str, default="ga", 
                        choices=["ga", "convblock", "convupblock", "attention"],
                        help="Target module to export")
    args = parser.parse_args()
    
    export_target(args.target)
