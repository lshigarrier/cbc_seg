import logging
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from utils import get_conf, RuntimeTracker
from data import ImageMaskDataset, image_mask_collate_fn

logging.basicConfig(level=logging.DEBUG, format='%(name)s:%(message)s')
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


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
        images, masks = batch
        # Masks shape should be (B, H, W) with class indices (0 to num_classes-1, or 255)

        logits = self.forward(images)

        # Combine losses
        loss_focal = self.focal_loss(logits, masks)
        loss_tversky = self.tversky_loss(logits, masks)
        total_loss = loss_focal + loss_tversky

        # Log the loss
        self.log(f"train_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
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
    main_logger = logging.getLogger('Train')
    conf = get_conf(main_logger)

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

    main_logger.info(f'Training on {len(train_set)} annotated images')

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
        callbacks=[checkpoint_callback, RuntimeTracker(), DeviceStatsMonitor()],
        max_epochs=conf.max_epochs,
        accelerator="gpu" if conf.use_gpu else "cpu",
        devices=1,
        precision="16-mixed",  # Automatic Mixed Precision (AMP)
        accumulate_grad_batches=conf.accumulate_grad_batches,  # Accumulate gradients over 4 batches
        log_every_n_steps=conf.log_every_n_steps  # Control logging verbosity
    )

    # model = torch.compile(model)
    trainer.fit(model, train_loader)

    '''
    # Load your trained model
    model = CBCSeg.load_from_checkpoint("path/to/best_model.ckpt")
    model.eval()

    # Create a dummy input tensor matching your patch size
    dummy_input = torch.randn(1, 3, conf.path_size, conf.path_size)

    # Export to ONNX
    model.to_onnx("cbc_seg.onnx", dummy_input, export_params=True)
    '''


if __name__ == "__main__":
    main()
