import logging
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

from utils import get_conf, logging_conf, pytorch_perf, RuntimeTracker
from data.data import ImageDataModule
# from models.deeplabv3plus import DeepLabV3Plus
from models.pidnet import PIDNet


def main():
    logging_conf()
    pytorch_perf()
    main_logger = logging.getLogger('Train')
    conf = get_conf(main_logger)
    pl.seed_everything(conf.seed, workers=True)

    datamodule = ImageDataModule(conf, main_logger)

    '''model = DeepLabV3Plus(
        lr=conf.lr,
        weight_decay=conf.weight_decay,
        eta_min=conf.eta_min,
        num_classes=conf.num_classes,
        tversky_alpha=conf.tversky_alpha,
        tversky_beta=conf.tversky_beta,
        ignore_index=conf.ignore_index
    )'''
    model = PIDNet(
        lr=conf.lr,
        weight_decay=conf.weight_decay,
        eta_min=conf.eta_min,
        num_classes=conf.num_classes,
        ignore_index=conf.ignore_index
    )

    logger = TensorBoardLogger(save_dir=conf.save_dir, name=conf.name)

    checkpoint_callback = ModelCheckpoint(
        monitor="train_loss",
        mode="min",
        save_top_k=1,
        filename="{epoch:02d}"
    )

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, RuntimeTracker()],
        max_epochs=conf.max_epochs,
        accelerator="gpu" if conf.use_gpu else "cpu",
        devices=1,
        precision="16-mixed",  # Automatic Mixed Precision (AMP)
        accumulate_grad_batches=conf.accumulate_grad_batches,
        log_every_n_steps=conf.log_every_n_steps,
        deterministic=conf.deterministic
    )

    trainer.fit(model, datamodule=datamodule)

    if conf.save_onnx:
        # Load your trained model
        # model = DeepLabV3Plus.load_from_checkpoint(checkpoint_callback.best_model_path)
        model = PIDNet.load_from_checkpoint(checkpoint_callback.best_model_path)
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
