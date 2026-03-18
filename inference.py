import logging
import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import get_conf, logging_conf, pytorch_perf, CustomTimer
from data import ImageDataset, image_collate_fn
from train import CBCSeg


def get_colormap(conf_colors):
    num_classes = len(conf_colors)
    cmap = np.zeros((num_classes, 4), dtype=np.uint8)

    # Background (Index 0) - Fully transparent
    cmap[0] = [conf_colors[0][0], conf_colors[0][1], conf_colors[0][2], 0]

    # All other classes - Solid color (opacity is handled later in your blending step)
    for i in range(1, num_classes):
        cmap[i] = [conf_colors[i][0], conf_colors[i][1], conf_colors[i][2], 255]

    return cmap


def save_legend(class_mapping, cmap, output_dir):
    patches = []
    sorted_classes = sorted(class_mapping.items(), key=lambda item: item[1])
    for class_name, class_idx in sorted_classes:
        color = cmap[class_idx][:3] / 255.0
        patches.append(mpatches.Patch(color=color, label=class_name))

    fig, ax = plt.subplots(figsize=(4, len(class_mapping) * 0.3))
    ax.legend(handles=patches, loc='center', frameon=False)
    ax.axis('off')

    legend_path = output_dir / 'legend.png'
    plt.savefig(legend_path, bbox_inches='tight', dpi=150)
    plt.close()


class CBCInferenceWrapper(CBCSeg):
    """Subclass CBCSeg just to add the predict step for this script"""

    def __init__(self, *args, output_dir=None, dataset=None, cmap=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.output_dir = output_dir
        self.dataset = dataset
        self.cmap = cmap

    def save_hyperparameters(self, *args, **kwargs):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        images, names, patches, boxes = batch

        # 1. Forward pass
        outputs = self(patches)

        # 2. Stitching
        logits = self.dataset.stitch(outputs, boxes)

        # 3. Processing & Saving
        for b in range(len(images)):
            mask_idx = torch.argmax(logits[b], dim=0).cpu().numpy()
            mask_rgba = self.cmap[mask_idx]

            mask_a = mask_rgba[:, :, 3:].astype(np.float32) / 255.0
            mask_rgb = mask_rgba[:, :, :3].astype(np.float32)

            image = images[b].cpu().numpy().transpose(1, 2, 0).astype(np.float32)
            overlay = image * (1 - mask_a * 0.5) + mask_rgb * (mask_a * 0.5)
            overlay = overlay.clip(0, 255)

            Image.fromarray(mask_rgb.astype(np.uint8)).save(self.output_dir / f'{names[b]}_mask.png')
            Image.fromarray(overlay.astype(np.uint8)).save(self.output_dir / f'{names[b]}_over.png')


def main():
    logging_conf()
    pytorch_perf()
    logger = logging.getLogger('Inference')
    conf = get_conf(logger)
    pl.seed_everything(conf.seed, workers=True)

    output_dir = Path(conf.save_dir) / conf.name / conf.version / 'result'
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(conf.save_dir) / conf.name / conf.version / 'checkpoints' / conf.ckpt

    cmap = get_colormap(conf.colors)
    save_legend(conf.class_mapping, cmap, output_dir)

    dataset = ImageDataset(
        conf.eval_data_dir,
        patch_per_row=conf.eval_patch_per_row,
        patch_per_col=conf.eval_patch_per_col,
        patch_size=conf.eval_patch_size,
        patch_overlap=conf.eval_patch_overlap
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=conf.num_workers,
        persistent_workers=True,
        pin_memory=conf.use_gpu,
        collate_fn=image_collate_fn
    )

    # Load model the wrapper that includes the predict_step
    model = CBCInferenceWrapper.load_from_checkpoint(
        ckpt_path,
        output_dir=output_dir,
        dataset=dataset,
        cmap=cmap
    )

    trainer = pl.Trainer(
        accelerator="gpu" if conf.use_gpu else "cpu",
        devices=1,
        precision="16-mixed",
        deterministic=conf.deterministic
    )

    timer = CustomTimer()
    timer.start()
    logger.info('-' * 70)
    logger.info(f'Inference on {len(dataset)} images')
    trainer.predict(model, dataloaders=dataloader)
    logger.info(f'Predictions saved in {output_dir}')
    timer.stop(logger, len(dataset))

    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    logger.info(f'Peak GPU memory allocated: {peak_memory_gb:.2f} GB')


if __name__ == '__main__':
    main()
