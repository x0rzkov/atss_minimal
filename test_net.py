import argparse
from ccc import update_config
from atss_core.data.data_loader import make_data_loader
from tqdm import tqdm
import torch
import os
from atss_core.data.coco_eval import do_coco_evaluation
from atss_core.utils.timer import Timer, get_time_str
from generalized_rcnn import GeneralizedRCNN
from atss_core.utils.checkpoint import Checkpointer

parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
parser.add_argument("--bs", type=int, default=4, help='test batch size.')
parser.add_argument("--weights", default="ATSS_R_50_FPN_1x.pth", help="path to the trained model")
parser.add_argument("--show_env", action='store_true', default=False, help="Whether to show the env information.")

args = parser.parse_args()
cfg = update_config(args)
cfg.print_cfg()

if cfg.show_env:
    from torch.utils.collect_env import get_pretty_env_info
    print(get_pretty_env_info())


def compute_on_dataset(model, data_loader, device, timer=None):
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        with torch.no_grad():
            if timer:
                timer.tic()
            output = model(images.to(device))
            if timer:
                torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        results_dict.update({img_id: result for img_id, result in zip(image_ids, output)})
    return results_dict


val_loader = make_data_loader(cfg, is_train=False)

model = GeneralizedRCNN(cfg).cuda()
model.eval()

checkpointer = Checkpointer(model, save_dir='.')
_ = checkpointer.load(cfg.weights)


dataset_name = 'coco_2017_val'

device = torch.device('cuda')
dataset = val_loader.dataset

total_timer = Timer()
inference_timer = Timer()
total_timer.tic()
predictions = compute_on_dataset(model, val_loader, device, inference_timer)

total_time = total_timer.toc()
total_time_str = get_time_str(total_time)

total_infer_time = get_time_str(inference_timer.total_time)

output_folder = 'results/coco_2017_val'
os.makedirs(output_folder, exist_ok=True)
torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

args = dict(dataset=dataset, predictions=predictions, output_folder=output_folder,
            iou_type='bbox', expected_results=[],expected_results_sigma_tol=4)

do_coco_evaluation(**args)
