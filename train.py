import logging
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pathlib import Path

from utils import get_conf, logging_conf, pytorch_perf, RuntimeTracker
from data import ImageMaskDataset, image_mask_collate_fn


class CBCSeg(pl.LightningModule):
    def __init__(
            self,
            num_classes=18,
            lr=3e-4,
            eta_min=1e-6,
            weight_decay=1e-4,
            tversky_alpha=0.3,
            tversky_beta=0.7,
            ignore_index=255
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = smp.DeepLabV3Plus(
            encoder_name="mit_b2",
            encoder_weights="imagenet",  # Pre-trained weights
            in_channels=3,
            classes=num_classes,
        )

        # Loss Functions: Combating Imbalance and enforcing Continuity
        # Focal Loss: Handles pixel-wise extreme class imbalance
        self.focal_loss = smp.losses.FocalLoss(
            mode=smp.losses.MULTICLASS_MODE,
            ignore_index=ignore_index
        )
        # Tversky Loss: Favors False Positives over False Negatives to close gaps
        self.tversky_loss = smp.losses.TverskyLoss(
            mode=smp.losses.MULTICLASS_MODE,
            alpha=tversky_alpha,
            beta=tversky_beta,
            ignore_index=ignore_index
        )

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        patches, masks = batch
        # Masks shape should be (B, H, W) with class indices (0 to num_classes-1, or 255)

        logits = self.forward(patches)

        # Combine losses
        loss_focal = self.focal_loss(logits, masks)
        loss_tversky = self.tversky_loss(logits, masks)
        total_loss = loss_focal + loss_tversky

        # Log the loss
        self.log(
            f"train_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=patches.shape[0]
        )

        return total_loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        # Smoothly decreases the learning rate to lr_min over max_epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.eta_min  # minimum learning rate
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",  # Update the learning rate after every epoch
                "frequency": 1
            },
        }


def main():
    logging_conf()
    pytorch_perf()
    main_logger = logging.getLogger('Train')
    conf = get_conf(main_logger)
    pl.seed_everything(conf.seed, workers=True)

    model = CBCSeg(
        num_classes=conf.num_classes,
        lr=conf.lr,
        eta_min=conf.eta_min,
        weight_decay=conf.weight_decay,
        ignore_index=conf.ignore_index)

    train_set = ImageMaskDataset(
        conf.data_dir,
        conf.patch_per_row,
        conf.patch_per_col,
        conf.patch_size,
        conf.patch_overlap)

    train_loader = DataLoader(
        train_set,
        batch_size=conf.batch_size,
        shuffle=True,
        num_workers=conf.num_workers,
        persistent_workers=True,
        pin_memory=conf.use_gpu,
        collate_fn=image_mask_collate_fn
    )

    # Define custom output directory
    logger = TensorBoardLogger(save_dir=conf.save_dir, name=conf.name)

    # Callbacks
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
        accumulate_grad_batches=conf.accumulate_grad_batches,  # Accumulate gradients over 4 batches
        log_every_n_steps=conf.log_every_n_steps,  # Control logging verbosity
        deterministic=conf.deterministic
    )

    main_logger.info('-' * 70)
    main_logger.info(f'Training on {len(train_set)} annotated images')
    trainer.fit(model, train_loader)

    if conf.save_onnx:
        # Load your trained model
        model = CBCSeg.load_from_checkpoint(checkpoint_callback.best_model_path)
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
