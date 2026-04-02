import logging
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pathlib import Path

from utils import get_conf, logging_conf, pytorch_perf, RuntimeTracker, CleanTensorBoardLogger
from data.data import ImageDataModule
from models.models import get_model


def main():
    logging_conf()
    pytorch_perf()
    main_logger = logging.getLogger('Train')
    conf = get_conf(main_logger)
    pl.seed_everything(conf.seed, workers=True)

    datamodule = ImageDataModule(conf, main_logger)

    model = get_model(task='train', conf=conf)

    logger = CleanTensorBoardLogger(save_dir=conf.save_dir, name=conf.name)

    best_checkpoint = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        filename="{epoch:03d}"
    )

    last_checkpoint = ModelCheckpoint(filename="last")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[best_checkpoint, last_checkpoint, RuntimeTracker()],
        max_epochs=conf.max_epochs,
        accelerator="gpu" if conf.use_gpu else "cpu",
        devices=1,
        precision="16-mixed",  # Automatic Mixed Precision (AMP)
        accumulate_grad_batches=conf.accumulate_grad_batches,
        log_every_n_steps=conf.log_every_n_steps,
        deterministic=conf.deterministic
    )

    ckpt_path = None
    if conf.from_ckpt:
        ckpt_path = Path(conf.save_dir) / conf.name / conf.version / 'checkpoints' / conf.ckpt

    trainer.fit(model, datamodule=datamodule, ckpt_path=ckpt_path)

    if conf.save_onnx:
        # Load your trained model
        model = get_model(task='load', conf=conf, ckpt_path=best_checkpoint.best_model_path)
        model.eval()

        dummy_input = torch.randn(1, 3, conf.patch_size, conf.patch_size)

        version_dir = trainer.logger.log_dir
        onnx_file_path = Path(version_dir) / 'cbc_seg.onnx'

        # Export to ONNX
        model.to_onnx(
            onnx_file_path,
            dummy_input,
            export_params=True,
            opset_version=14,  # Good default for modern PyTorch/ONNX
            input_names=['input'],
            output_names=['output']
        )
        main_logger.info("Training and ONNX export complete!")

    else:
        main_logger.info("Training complete!")

    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    main_logger.info(f'Peak GPU memory allocated: {peak_memory_gb:.2f} GB')


if __name__ == "__main__":
    main()
