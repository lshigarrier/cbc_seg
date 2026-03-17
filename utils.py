import sys
import yaml
import time
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
    return conf


class RuntimeTracker(Callback):
    def on_train_start(self, trainer, pl_module):
        # Record the exact time training starts
        self.train_start_time = time.time()

    def on_train_epoch_start(self, trainer, pl_module):
        # Record the start of each epoch
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        # Calculate durations
        epoch_time = time.time() - self.epoch_start_time
        total_time = time.time() - self.train_start_time

        # Log them to TensorBoard
        # Using pl_module.log sends the data directly to your TensorBoardLogger
        pl_module.log("runtime/epoch_time_seconds", epoch_time)
        pl_module.log("runtime/total_time_seconds", total_time)
