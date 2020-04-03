import argparse
import cv2
import os

from ccc import update_config
from demo.predictor import COCODemo
import pdb
import time

parser = argparse.ArgumentParser(description="PyTorch Object Detection Webcam Demo")
parser.add_argument("--weights", default="res50_1x.pth", help="path to the trained model")
parser.add_argument("--images", default="images", help="path to demo images directory")

args = parser.parse_args()
cfg = update_config(args)
cfg.print_cfg()

# prepare object that handles inference plus adds predictions on top of image
coco_demo = COCODemo(cfg, thre_per_classes=cfg.thre_per_classes, min_img_size=cfg.min_img_size)

for im_name in os.listdir(args.images):
    img = cv2.imread(os.path.join(args.images, im_name))
    start_time = time.time()
    composite = coco_demo.run_on_opencv_image(img)
    print("{}\tinference time: {:.2f}s".format(im_name, time.time() - start_time))
    cv2.imshow(im_name, composite)
print("Press any keys to exit ...")
cv2.waitKey()
cv2.destroyAllWindows()
