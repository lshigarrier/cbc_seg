import segmentation_models_pytorch as smp

from models.cbcseg import CBCSeg


class DeepLabV3Plus(CBCSeg):
    def __init__(
            self,
            *args,
            num_classes=19,
            tversky_alpha=0.3,
            tversky_beta=0.7,
            ignore_index=255,
            **kwargs
    ):
        super().__init__(*args, **kwargs)

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
