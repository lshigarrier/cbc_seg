"""
Script to export a trained PIDNet model to TorchScript format.
This generates a standalone .pt file containing both the architecture and weights,
ready for inference without requiring the original source code.

Inputs:
    Hardcoded variables in the main() function:
    - ckpt_path: Path to the PyTorch Lightning checkpoint (.ckpt).
    - output_dir: Directory where the TorchScript model will be saved.
    - patch_size: Size of the square patch used for the dummy input.

Outputs:
    A TorchScript file named 'pidnet_traced.pt' saved in the specified output_dir.

Example command:
    python export_torchscript.py
"""

import logging
from pathlib import Path
import torch

from models.pidnet import PIDNet
from utils import logging_conf, pytorch_perf


def export_model_to_torchscript(ckpt_path: Path, output_path: Path, patch_size: int, logger: logging.Logger) -> None:
    """
    Loads the PIDNet model from a checkpoint, traces its execution graph,
    and saves it as a TorchScript file.
    """
    logger.info(f"Loading PIDNet from checkpoint: {ckpt_path}")

    # Load the model directly using PyTorch Lightning's built-in method.
    # Default arguments in PIDNet __init__ are used.
    model = PIDNet.load_from_checkpoint(ckpt_path)

    # Set to evaluation mode and move to CPU for generic compatibility
    model.eval()
    model.to('cpu')

    dummy_input = torch.randn(1, 3, patch_size, patch_size, device='cpu')
    traced_model = torch.jit.trace(model.model, dummy_input)

    logger.info(f"Saving traced model to: {output_path}")
    traced_model.save(str(output_path))
    logger.info("TorchScript export completed successfully.")


def main():
    logging_conf()
    pytorch_perf()
    logger = logging.getLogger('Export')

    ckpt_path = Path("C:/Users/Admin/Code/cbc_seg/output/pidnet_l/version_1/checkpoints/epoch=973.ckpt")
    output_dir = Path("C:/Users/Admin/Code/cbc_seg/cbc-pidnet-large")
    file_name = "cbc-pidnet-large.pt"
    patch_size = 1024

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / file_name

    export_model_to_torchscript(ckpt_path, output_path, patch_size, logger)


if __name__ == "__main__":
    main()
