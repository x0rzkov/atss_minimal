import argparse
from ccc import update_config
from atss_core.data.data_loader import make_data_loader
from tqdm import tqdm
import torch
import os
from atss_core.data.coco_eval import do_coco_evaluation
from generalized_rcnn import GeneralizedRCNN
import pdb

parser = argparse.ArgumentParser(description="PyTorch Object Detection Inference")
parser.add_argument("--bs", type=int, default=4, help='test batch size.')
parser.add_argument("--weights", default="res50_1x.pth", help="path to the trained model")
parser.add_argument("--show_env", action='store_true', default=False, help="Whether to show the env information.")

args = parser.parse_args()
cfg = update_config(args)
cfg.print_cfg()

if cfg.show_env:
    from torch.utils.collect_env import get_pretty_env_info
    print(get_pretty_env_info())

val_loader = make_data_loader(cfg, is_train=False)

model = GeneralizedRCNN(cfg).cuda()
model.eval()

checkpoint = torch.load(cfg.weights, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint)

predictions = {}

with torch.no_grad():
    for _, (images, targets, image_ids) in enumerate(tqdm(val_loader)):
        output = model(images.to(torch.device('cuda')))
        output = output[0].to(torch.device('cpu'))
        pdb.set_trace()
        predictions.update({img_id: result for img_id, result in zip(image_ids, output)})


output_folder = 'results/coco_2017_val'
os.makedirs(output_folder, exist_ok=True)
torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

do_coco_evaluation(dataset=val_loader.dataset, predictions=predictions, output_folder=output_folder,
                   iou_type='bbox', expected_results=[], expected_results_sigma_tol=4)
