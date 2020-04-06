import argparse
import cv2
import os
import torch
from ccc import update_config
from atss_core.augmentations import detect_aug
from generalized_rcnn import GeneralizedRCNN
from atss_core.structures.image_list import to_image_list
import pdb
import time

parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
parser.add_argument("--weights", default="res50_1x.pth", help="path to the trained model")
parser.add_argument("--images", default="images", help="path to demo images directory")

args = parser.parse_args()
cfg = update_config(args)
cfg.print_cfg()

model = GeneralizedRCNN(cfg).cuda()
checkpoint = torch.load(cfg.weights, map_location=torch.device("cpu"))
model.load_state_dict(checkpoint)
model.eval()

transforms = detect_aug(cfg)

palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

thre_per_classes = torch.tensor(cfg.thre_per_classes)

for im_name in os.listdir(args.images):
    img = cv2.imread(os.path.join(args.images, im_name))
    start_time = time.time()

    height, width = img.shape[:-1]
    img_aug = transforms(img)

    # convert to an ImageList, padded so that it is divisible by cfg.DATALOADER.SIZE_DIVISIBILITY
    image_list = to_image_list(img_aug, cfg.size_divisibility)
    image_list = image_list.to(torch.device('cuda'))

    with torch.no_grad():
        predictions = model(image_list)[0].to(torch.device('cpu'))
    predictions = predictions.resize((width, height))

    scores = predictions.get_field("scores")
    labels = predictions.get_field("labels")
    thresholds = thre_per_classes[(labels - 1).long()]
    keep = torch.nonzero(scores > thresholds).squeeze(1)
    predictions = predictions[keep]
    scores = predictions.get_field("scores")
    _, idx = scores.sort(0, descending=True)

    top_predictions = predictions[idx]

    labels = top_predictions.get_field("labels")
    boxes = top_predictions.bbox

    colors = labels[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8").tolist()

    for box, color in zip(boxes, colors):
        box = box.to(torch.int64)
        top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
        cv2.rectangle(img, tuple(top_left), tuple(bottom_right), tuple(color), 2)

    scores = top_predictions.get_field("scores").tolist()
    labels = top_predictions.get_field("labels").tolist()
    labels = [cfg.class_names[i] for i in labels]
    boxes = top_predictions.bbox

    for box, score, label in zip(boxes, scores, labels):
        x, y = box[:2]
        cv2.putText(img, f'{label}: {score:.2f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

    print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
    cv2.imshow(im_name, img)

print("Press any keys to exit ...")
cv2.waitKey()
cv2.destroyAllWindows()
