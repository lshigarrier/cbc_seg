"""
Script to export a trained PIDNet model to ONNX format.
This generates a standalone .onnx file containing both the architecture and weights,
ready for inference without requiring the original source code or PyTorch.

Inputs:
    Hardcoded variables in the main() function:
    - ckpt_path: Path to the PyTorch Lightning checkpoint (.ckpt).
    - output_dir: Directory where the ONNX model will be saved.
    - patch_size: Size of the square patch used for the dummy input.

Outputs:
    A TorchScript file named 'cbc-pidnet-large.onnx' saved in the specified output_dir.

Example command:
    python export_onnx.py
"""

import logging
from pathlib import Path
import torch

from models.pidnet import PIDNet
from utils import logging_conf, pytorch_perf


def export_model_to_onnx(ckpt_path: Path, output_path: Path, patch_size: int, logger: logging.Logger) -> None:
    """
    Loads the PIDNet model from a checkpoint and exports it to ONNX format with a dynamic batch size.
    """
    logger.info(f"Loading PIDNet from checkpoint: {ckpt_path}")

    # Load the model directly using PyTorch Lightning's built-in method.
    # Default arguments in PIDNet __init__ are used.
    model = PIDNet.load_from_checkpoint(ckpt_path)

    # Set to evaluation mode and move to CPU for generic compatibility
    model.eval()
    model.to('cpu')

    # Dummy input with batch_size = 1 for the export trace
    dummy_input = torch.randn(1, 3, patch_size, patch_size, device='cpu')

    logger.info(f"Exporting model to ONNX: {output_path}")
    # Export the model
    torch.onnx.export(
        model.model,  # The underlying PyTorch nn.Module
        (dummy_input,),  # Dummy model input
        str(output_path),  # Where to save the model
        export_params=True,  # Store the trained parameter weights inside the model file
        opset_version=13,  # The ONNX version to export the model to (13 is very stable)
        do_constant_folding=True,  # Execute constant folding for optimization
        input_names=['input'],  # The model's input names (useful for onnxruntime later)
        output_names=['output'],  # The model's output names
        dynamic_axes={  # Variable length axes (dynamic batch size)
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    logger.info("ONNX export completed successfully.")


def main():
    logging_conf()
    pytorch_perf()
    logger = logging.getLogger('Export')

    ckpt_path = Path("C:/Users/Admin/Code/cbc_seg/output/pidnet_l/version_1/checkpoints/epoch=973.ckpt")
    output_dir = Path("C:/Users/Admin/Code/cbc_seg/cbc-pidnet-large")
    file_name = "cbc-pidnet-large.onnx"
    patch_size = 1024

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name

    export_model_to_onnx(ckpt_path, output_path, patch_size, logger)


if __name__ == "__main__":
    main()
