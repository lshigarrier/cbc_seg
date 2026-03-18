import sys
import yaml
import time
import logging
import torch
from pytorch_lightning.callbacks import Callback
from types import SimpleNamespace
from pathlib import Path


def get_conf(logger):
    filename = sys.argv[1]
    config_path = Path("config") / f"{filename}.yaml"
    conf = yaml.safe_load(config_path.read_text())
    conf = SimpleNamespace(**conf)
    logger.info('-' * 70)
    for key, value in vars(conf).items():
        logger.info(f"{key} : {value}")
    logger.info('-' * 70)
    return conf


def logging_conf():
    logging.basicConfig(level=logging.INFO, format='%(name)s:%(message)s')


def pytorch_perf():
    torch.set_float32_matmul_precision('high')
    torch.backends.cudnn.benchmark = True


class CustomTimer:

    def __init__(self):
        self.start_perf_count = None
        self.start_process_time = None

    def start(self):
        self.start_perf_count = time.perf_counter()
        self.start_process_time = time.process_time()

    def stop(self, logger, len_dataset):
        elapsed_perf_count = time.perf_counter() - self.start_perf_count
        elapsed_process_time = time.process_time() - self.start_process_time
        logger.info(f"Perf counter: {elapsed_perf_count:.2f} s")
        logger.info(f"  Time per image: {elapsed_perf_count / len_dataset * 1000:.2f} ms")
        logger.info(f"Process time: {elapsed_process_time:.2f} s")
        logger.info(f"  Time per image: {elapsed_process_time / len_dataset * 1000:.2f} ms")


class RuntimeTracker(Callback):

    def __init__(self):
        self.total_start_perf_count = None
        self.total_start_process_time = None
        self.epoch_start_perf_count = None
        self.epoch_start_process_time = None

    def on_train_start(self, trainer, pl_module):
        # Record the exact time training starts
        self.total_start_perf_count = time.perf_counter()
        self.total_start_process_time = time.process_time()

    def on_train_epoch_start(self, trainer, pl_module):
        # Record the start of each epoch
        self.epoch_start_perf_count = time.perf_counter()
        self.epoch_start_process_time = time.process_time()

    def on_train_epoch_end(self, trainer, pl_module):
        # Calculate durations
        epoch_perf_count = time.perf_counter() - self.epoch_start_perf_count
        total_perf_count = time.perf_counter() - self.total_start_perf_count
        epoch_process_time = time.process_time() - self.epoch_start_process_time
        total_process_time = time.process_time() - self.total_start_process_time

        # Log them to TensorBoard
        # Using pl_module.log sends the data directly to the TensorBoardLogger
        pl_module.log("runtime/epoch_perf_counter_seconds", epoch_perf_count)
        pl_module.log("runtime/total_perf_counter_seconds", total_perf_count)
        pl_module.log("runtime/epoch_process_time_seconds", epoch_process_time)
        pl_module.log("runtime/total_process_time_seconds", total_process_time)
