import argparse
import os
import time
import torch
import datetime
from ccc import update_config
from atss_core.data.data_loader import make_data_loader
from atss_core.solver.lr_scheduler import WarmupMultiStepLR
from atss_core.solver.optimizer import make_optimizer
from atss_core.utils.metric_logger import MetricLogger
from generalized_rcnn import GeneralizedRCNN
from atss_core.utils.checkpoint import Checkpointer


parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
parser.add_argument("--bs", type=int, default=4, help='test batch size.')
parser.add_argument("--show_env", action='store_true', default=False, help="Whether to show the env information.")

args = parser.parse_args()
cfg = update_config(args)
cfg.print_cfg()

if cfg.show_env:
    from torch.utils.collect_env import get_pretty_env_info

    print(get_pretty_env_info())

model = GeneralizedRCNN(cfg).cuda()

optimizer = make_optimizer(cfg, model)
scheduler = WarmupMultiStepLR(optimizer, (60000, 80000), 0.1, warmup_factor=0.33333333, warmup_iters=500,
                              warmup_method='constant')

arguments = {}
arguments["iteration"] = 0

output_dir = cfg.OUTPUT_DIR

checkpointer = Checkpointer(model, optimizer, scheduler, output_dir)

extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT)
arguments.update(extra_checkpoint_data)

data_loader = make_data_loader(cfg, is_train=True, start_iter=arguments["iteration"])

checkpoint_period = 2500

meters = MetricLogger(delimiter="  ")
max_iter = len(data_loader)
start_iter = arguments["iteration"]
model.train()
start_training_time = time.time()
end = time.time()


for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
    data_time = time.time() - end
    iteration = iteration + 1
    arguments["iteration"] = iteration

    images = images.cuda()
    targets = [target.cuda() for target in targets]

    loss_dict = model(images, targets)

    losses = sum(loss for loss in loss_dict.values())

    losses_reduced = sum(loss for loss in loss_dict.values())
    meters.update(loss=losses_reduced, **loss_dict)

    optimizer.zero_grad()
    losses.backward()
    optimizer.step()

    # scheduler.step() should be run after optimizer.step()
    scheduler.step()

    batch_time = time.time() - end
    end = time.time()
    meters.update(time=batch_time, data=data_time)

    eta_seconds = meters.time.global_avg * (max_iter - iteration)
    eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

    if iteration % 20 == 0 or iteration == max_iter:
        pass
        # logger.info(
        #     meters.delimiter.join(
        #         [
        #             "eta: {eta}",
        #             "iter: {iter}",
        #             "{meters}",
        #             "lr: {lr:.6f}",
        #             "max mem: {memory:.0f}",
        #         ]
        #     ).format(
        #         eta=eta_string,
        #         iter=iteration,
        #         meters=str(meters),
        #         lr=optimizer.param_groups[0]["lr"],
        #         memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
        #     )
        # )
    if iteration % checkpoint_period == 0:
        checkpointer.save("model_{:07d}".format(iteration), **arguments)
    if iteration == max_iter:
        checkpointer.save("model_final", **arguments)

total_training_time = time.time() - start_training_time
total_time_str = str(datetime.timedelta(seconds=total_training_time))
