import logging
import torch
import numpy as np
import pytorch_lightning as pl
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from utils import get_conf, logging_conf, pytorch_perf, CustomTimer
from data.data import ImageDataModule
# from models.deeplabv3plus import DeepLabV3Plus
from models.pidnet import PIDNet


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


def main():
    logging_conf()
    pytorch_perf()
    logger = logging.getLogger('Inference')
    conf = get_conf(logger)
    pl.seed_everything(conf.seed, workers=True)

    output_dir = Path(conf.save_dir) / conf.name / conf.version / conf.result_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(conf.save_dir) / conf.name / conf.version / 'checkpoints' / conf.ckpt

    cmap = get_colormap(conf.colors)
    save_legend(conf.class_mapping, cmap, output_dir)

    datamodule = ImageDataModule(conf, logger)

    '''model = DeepLabV3Plus.load_from_checkpoint(
        ckpt_path,
        num_classes=conf.num_classes,
        patch_per_img=conf.patch_per_row*conf.patch_per_col,
        output_dir=output_dir,
        cmap=cmap
    )'''
    model = PIDNet.load_from_checkpoint(
        ckpt_path,
        num_classes=conf.num_classes,
        patch_per_img=conf.patch_per_row*conf.patch_per_col,
        output_dir=output_dir,
        cmap=cmap
    )

    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu" if conf.use_gpu else "cpu",
        devices=1,
        precision="16-mixed",
        deterministic=conf.deterministic
    )

    timer = CustomTimer()
    timer.start()
    trainer.predict(model, datamodule=datamodule)
    logger.info(f'Predictions saved in {output_dir}')
    timer.stop(logger, len(datamodule.predict_dataset))

    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    logger.info(f'Peak GPU memory allocated: {peak_memory_gb:.2f} GB')


if __name__ == '__main__':
    main()
