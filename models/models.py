from models.deeplabv3plus import DeepLabV3Plus
from models.pidnet import PIDNet


def get_model(task, conf, ckpt_path=None, output_dir=None, cmap=None):
    """
    :param conf:
    :param task:
        'train' -> training
        'inference' -> inference
        'load' -> load from checkpoint
    :param ckpt_path:
    :param output_dir:
    :param cmap:
    :return:
    """
    if task == 'train':
        if conf.name == 'deeplabv3+_mitb2':
            model = DeepLabV3Plus(
                lr=conf.lr,
                weight_decay=conf.weight_decay,
                eta_min=conf.eta_min,
                num_classes=conf.num_classes,
                tversky_alpha=conf.tversky_alpha,
                tversky_beta=conf.tversky_beta,
                ignore_index=conf.ignore_index
            )
        elif conf.name == 'pidnet_l':
            model = PIDNet(
                lr=conf.lr,
                weight_decay=conf.weight_decay,
                eta_min=conf.eta_min,
                num_classes=conf.num_classes,
                ignore_index=conf.ignore_index
            )
        else:
            raise NotImplementedError
    elif task == 'inference':
        if conf.name == 'deeplabv3+_mitb2':
            model = DeepLabV3Plus.load_from_checkpoint(
                ckpt_path,
                num_classes=conf.num_classes,
                patch_per_img=conf.patch_per_row*conf.patch_per_col,
                output_dir=output_dir,
                cmap=cmap
            )
        elif conf.name == 'pidnet_l':
            model = PIDNet.load_from_checkpoint(
                ckpt_path,
                num_classes=conf.num_classes,
                patch_per_img=conf.patch_per_row*conf.patch_per_col,
                output_dir=output_dir,
                cmap=cmap
            )
        else:
            raise NotImplementedError
    elif task == 'load':
        if conf.name == 'deeplabv3+_mitb2':
            model = DeepLabV3Plus.load_from_checkpoint(ckpt_path)
        elif conf.name == 'pidnet_l':
            model = PIDNet.load_from_checkpoint(ckpt_path)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError
    return model
