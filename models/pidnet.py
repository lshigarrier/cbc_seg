from mmseg.registry import MODELS
from mmseg.structures import SegDataSample
from mmseg.utils import register_all_modules
from mmengine.config import ConfigDict
from mmengine.runner import load_checkpoint
from mmengine.structures import PixelData

from models.cbcseg import CBCSeg


class PIDNet(CBCSeg):
    """
    https://github.com/open-mmlab/mmsegmentation/tree/main/configs/pidnet
    """
    def __init__(
            self,
            *args,
            num_classes=19,
            ignore_index=255,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.ignore_index = ignore_index

        # MMSegmentation configuration for PIDNet-Large
        model_cfg = ConfigDict(
            type='EncoderDecoder',
            data_preprocessor=dict(
                type='SegDataPreProcessor',
                mean=None,
                std=None,
                bgr_to_rgb=False,
                pad_val=0,
                seg_pad_val=ignore_index,
                size=None
            ),
            backbone=dict(
                type='PIDNet',
                in_channels=3,
                channels=64,
                ppm_channels=112,
                num_stem_blocks=3,
                num_branch_blocks=4,
                align_corners=False,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True)
            ),
            decode_head=dict(
                type='PIDHead',
                in_channels=256,
                channels=256,
                num_classes=num_classes,
                ignore_index=ignore_index,
                align_corners=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                act_cfg=dict(type='ReLU', inplace=True),
                # MMSegmentation handles the 3 branches and losses automatically!
                loss_decode=[
                    dict(
                        type='CrossEntropyLoss',
                        use_sigmoid=False,
                        loss_weight=0.4,
                        avg_non_ignore=True
                    ),
                    dict(
                        type='OhemCrossEntropy',
                        thres=0.9,
                        min_kept=131072,
                        loss_weight=1.0
                    ),
                    dict(type='BoundaryLoss', loss_weight=20.0),
                    dict(
                        type='OhemCrossEntropy',
                        thres=0.9,
                        min_kept=131072,
                        loss_weight=1.0
                    )
                ]
            ),
            train_cfg=dict(),
            test_cfg=dict(mode='whole')
        )

        # Build the model using MMSegmentation's registry
        register_all_modules(init_default_scope=True)
        self.model = MODELS.build(model_cfg)

        # Load MMSegmentation official pre-trained PIDNet-Large (Cityscapes)
        checkpoint_url = "https://download.openmmlab.com/mmsegmentation/v0.5/pidnet/pidnet-l_2xb6-120k_1024x1024-cityscapes/pidnet-l_2xb6-120k_1024x1024-cityscapes_20230303_114514-0783ca6b.pth"
        load_checkpoint(self.model, checkpoint_url, map_location='cpu', strict=False)

    def forward(self, x):
        return self.model(inputs=x, mode='tensor')

    def training_step(self, batch, batch_idx):
        patches, masks = batch

        # Since you use 255 for boundary dilation, we can extract the boundary target directly
        # 1.0 for boundary, 0.0 for non-boundary
        boundary_targets = (masks == self.ignore_index).float()

        # MMSegmentation expects data in a specific dictionary format for training
        data = {
            'inputs': patches,
            'data_samples': []
        }

        for i in range(patches.size(0)):
            data_sample = SegDataSample()
            data_sample.gt_sem_seg = PixelData(data=masks[i].unsqueeze(0))
            data_sample.gt_edge_map = PixelData(data=boundary_targets[i].unsqueeze(0))
            data['data_samples'].append(data_sample)

        # When calling forward with 'loss' mode, MMSegmentation automatically computes
        # the OhemCrossEntropy and Boundary losses defined in the config.
        loss_dict = self.model(inputs=data['inputs'], data_samples=data['data_samples'], mode='loss')
        total_loss, log_vars = self.model.parse_losses(loss_dict)

        # Log the loss
        self.log(
            "train_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=patches.shape[0]
        )

        return total_loss
