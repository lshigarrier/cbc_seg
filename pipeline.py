import logging
import torch
import pytorch_lightning as pl

from utils import get_conf, logging_conf, pytorch_perf, CustomTimer
from data.data import ImageDataModule
from models.models import get_model


def run_inference(conf, logger):
    datamodule = ImageDataModule(conf, logger)
    output_dir = conf.save_dir / "predictions"
    model = get_model(
        task='inference',
        conf=conf,
        ckpt_path=conf.ckpt_path,
        output_dir=output_dir
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
    timer.stop(logger, len_dataset=len(datamodule.predict_dataset))

    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    logger.info(f'Peak GPU memory allocated: {peak_memory_gb:.2f} GB')


def run_mosaic(conf, logger):
    pass


def run_postprocessing(conf, logger):
    pass


def main():
    # Initialization
    logging_conf()
    pytorch_perf()
    logger = logging.getLogger('Pipeline')
    conf = get_conf(logger)
    pl.seed_everything(conf.seed, workers=True)
    # Inference
    if conf.inference_flag:
        run_inference(conf, logger)
    # Mosaic
    if conf.mosaic_flag:
        run_mosaic(conf, logger)
    # Postprocessing
    if conf.postprocessing_flag:
        run_postprocessing(conf, logger)


if __name__ == '__main__':
    main()
