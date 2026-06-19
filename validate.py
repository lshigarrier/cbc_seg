import logging
import torch
import pytorch_lightning as pl
from pathlib import Path

from utils import get_conf, logging_conf, pytorch_perf, CustomTimer
from data.data import ImageDataModule
from models.models import get_model


def main():
    logging_conf()
    pytorch_perf()
    logger = logging.getLogger('Validate')
    conf = get_conf(logger)
    pl.seed_everything(conf.seed, workers=True)

    ckpt_path = Path(conf.save_dir) / conf.name / conf.version / 'checkpoints' / conf.ckpt

    datamodule = ImageDataModule(conf, logger)

    model = get_model(task='validate', conf=conf, ckpt_path=ckpt_path, logger=logger)

    trainer = pl.Trainer(
        logger=False,
        accelerator="gpu" if conf.use_gpu else "cpu",
        devices=1,
        precision="16-mixed",
        deterministic=conf.deterministic
    )

    timer = CustomTimer()
    timer.start()
    trainer.validate(model, datamodule=datamodule)
    logger.info('Validation done')
    timer.stop(logger, len_dataset=len(datamodule.val_dataset))

    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    logger.info(f'Peak GPU memory allocated: {peak_memory_gb:.2f} GB')


if __name__ == '__main__':
    main()
