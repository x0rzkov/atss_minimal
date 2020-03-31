import logging
import os
import pdb
import torch
from atss_core.utils.model_serialization import load_state_dict


class Checkpointer(object):
    def __init__(self, model, optimizer=None, scheduler=None, save_dir="", logger=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.save_dir = save_dir
        if logger is None:
            logger = logging.getLogger(__name__)
        self.logger = logger

    def save(self, name, **kwargs):
        if not self.save_dir:
            return

        data = {}
        data.setdefault('model', self.model.state_dict())
        data.setdefault('optimizer', self.optimizer.state_dict())
        data.setdefault('scheduler', self.scheduler.state_dict())
        data.update(kwargs)

        save_file = os.path.join(self.save_dir, "{}.pth".format(name))
        self.logger.info("Saving checkpoint to {}".format(save_file))
        torch.save(data, save_file)

    def load(self, f=None):
        self.logger.info(f"Loading checkpoint from {f}")
        checkpoint = torch.load(f, map_location=torch.device("cpu"), )
        load_state_dict(self.model, checkpoint.pop("model"))

        if "optimizer" in checkpoint and self.optimizer:
            self.logger.info("Loading optimizer from {}".format(f))
            self.optimizer.load_state_dict(checkpoint.pop("optimizer"))
        if "scheduler" in checkpoint and self.scheduler:
            self.logger.info("Loading scheduler from {}".format(f))
            self.scheduler.load_state_dict(checkpoint.pop("scheduler"))

        # return any further checkpoint data
        return checkpoint
