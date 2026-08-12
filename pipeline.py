import logging
import torch
import os
import pytorch_lightning as pl
from pathlib import Path

from utils import get_one_conf, logging_conf, pytorch_perf, CustomTimer
from data.data import ImageDataModule
from models.models import get_model
from positioning import process_passage
from utils_positioning import process_detections, generate_global_cog


def run_inference(conf, logger):
    conf.eval_data_dir = conf.save_dir
    datamodule = ImageDataModule(conf, logger)
    output_dir = conf.save_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)
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

    trainer.predict(model, datamodule=datamodule)
    logger.info(f'Predictions saved in {output_dir}')

    peak_memory_gb = torch.cuda.max_memory_allocated() / 1024 ** 3
    logger.info(f'Peak GPU memory allocated: {peak_memory_gb:.2f} GB')

    return len(datamodule.predict_dataset)


def get_leaf_jpg_directories(current_dir: Path) -> list[Path]:
    valid_leaves = []
    for dirpath, dirnames, filenames in os.walk(current_dir):
        if not dirnames:
            # Check if at least one file has a .jpg or .JPG extension
            if any(f.lower().endswith('.jpg') for f in filenames):
                valid_leaves.append(Path(dirpath))
    return valid_leaves


def run_positioning(conf, logger):
    paths_file = conf.save_dir / "paths.txt"
    passage_dirs = []
    with paths_file.open('r', encoding='utf-8') as file:
        for line in file:
            cleaned_path_str = line.strip().strip('"').strip("'")
            if not cleaned_path_str:
                continue
            current_dir = Path(cleaned_path_str)
            leaf_dirs = get_leaf_jpg_directories(current_dir)
            passage_dirs.extend(leaf_dirs)

    all_passage_data = []

    for passage_dir in passage_dirs:
        passage_name = passage_dir.name
        logger.info(f"  Processing passage {passage_name}")
        df_coords = process_passage(conf, passage_dir, logger)
        if df_coords is None or df_coords.empty:
            logger.error(f"  Failed to process passage {passage_name}")
            continue
        all_passage_data.append((df_coords, passage_dir))

    conf.detection_dir = conf.save_dir / "predictions"
    process_detections(conf, all_passage_data, conf.save_dir, logger)

    generate_global_cog(conf, all_passage_data, conf.save_dir, logger)

    return sum(1 for entry in os.scandir(conf.detection_dir) if entry.is_file())


def run_postprocessing(conf, logger):
    return 0


def main():
    # Initialization
    logging_conf()
    pytorch_perf()
    logger = logging.getLogger('Pipeline')
    conf = get_one_conf(logger)
    conf.save_dir = Path(conf.save_dir)
    pl.seed_everything(conf.seed, workers=True)

    timer = CustomTimer()
    timer.start()

    len_dataset = None
    # Inference
    if conf.inference_flag:
        len_dataset = run_inference(conf, logger)
    # Positioning
    if conf.positioning_flag:
        len_dataset = run_positioning(conf, logger)
    # Postprocessing
    if conf.postprocessing_flag:
        len_dataset = run_postprocessing(conf, logger)

    timer.stop(logger, len_dataset=len_dataset, show_time_per_image=len_dataset)


if __name__ == '__main__':
    main()
