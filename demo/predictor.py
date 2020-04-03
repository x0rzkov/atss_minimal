import cv2
import torch
from generalized_rcnn import GeneralizedRCNN
from atss_core.structures.image_list import to_image_list
from atss_core.augmentations import detect_aug
import pdb


class COCODemo:
    def __init__(self, cfg, thre_per_classes, min_img_size=800):
        self.cfg = cfg

        self.model = GeneralizedRCNN(cfg).cuda()
        self.model.eval()
        self.min_image_size = min_img_size

        checkpoint = torch.load(cfg.weights, map_location=torch.device("cpu"))
        self.model.load_state_dict(checkpoint)

        self.transforms = detect_aug(cfg)

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.thre_per_classes = torch.tensor(thre_per_classes)

    def run_on_opencv_image(self, original_image):
        height, width = original_image.shape[:-1]
        image = self.transforms(original_image)

        # convert to an ImageList, padded so that it is divisible by cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.size_divisibility)
        image_list = image_list.to(torch.device('cuda'))

        with torch.no_grad():
            predictions = self.model(image_list)[0].to(torch.device('cpu'))
        predictions = predictions.resize((width, height))

        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels")
        thresholds = self.thre_per_classes[(labels - 1).long()]
        keep = torch.nonzero(scores > thresholds).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)

        top_predictions = predictions[idx]

        result = original_image.copy()

        labels = top_predictions.get_field("labels")
        boxes = top_predictions.bbox

        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8").tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            result = cv2.rectangle(result, tuple(top_left), tuple(bottom_right), tuple(color), 2)

        scores = top_predictions.get_field("scores").tolist()
        labels = top_predictions.get_field("labels").tolist()
        labels = [self.cfg.class_names[i] for i in labels]
        boxes = top_predictions.bbox

        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            cv2.putText(result, f'{label}: {score:.2f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        return result
