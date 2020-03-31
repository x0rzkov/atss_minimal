import cv2
import torch
from torchvision import transforms as T
from generalized_rcnn import GeneralizedRCNN
from atss_core.utils.checkpoint import Checkpointer
from atss_core.structures.image_list import to_image_list
import pdb


class COCODemo:
    def __init__(self, cfg, thre_per_classes, min_img_size=800):
        self.cfg = cfg

        self.model = GeneralizedRCNN(cfg).cuda()
        self.model.eval()
        self.min_image_size = min_img_size

        checkpointer = Checkpointer(self.model, save_dir='.')
        _ = checkpointer.load(cfg.weights)

        self.transforms = self.build_transform()

        # used to make colors for each class
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])

        self.cpu_device = torch.device("cpu")
        self.thre_per_classes = torch.tensor(thre_per_classes)

    def build_transform(self):
        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        to_bgr_transform = T.Lambda(lambda x: x * 255)
        normalize_transform = T.Normalize(mean=self.cfg.pixel_mean, std=self.cfg.pixel_std)

        transform = T.Compose([T.ToPILImage(),
                               T.Resize(self.min_image_size),
                               T.ToTensor(),
                               to_bgr_transform,
                               normalize_transform])
        return transform

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        predictions = self.compute_prediction(image)
        top_predictions = self.select_top_predictions(predictions)

        result = image.copy()

        result = self.draw_boxes(result, top_predictions)
        result = self.draw_class_names(result, top_predictions)

        return result

    def compute_prediction(self, original_img):
        """
        Arguments:
            original_img (np.ndarray): an image as returned by OpenCV

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        # apply pre-processing to image
        image = self.transforms(original_img)
        # convert to an ImageList, padded so that it is divisible by cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.size_divisibility)
        image_list = image_list.to(torch.device('cuda'))

        with torch.no_grad():
            predictions = self.model(image_list)

        predictions = [aa.to(torch.device('cpu')) for aa in predictions]

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_img.shape[:-1]
        prediction = prediction.resize((width, height))

        return prediction

    def select_top_predictions(self, predictions):
        """
        Select only predictions which have a `score` > self.confidence_threshold,
        and returns the predictions in descending order of score

        Arguments:
            predictions (BoxList): the result of the computation by the model.
                It should contain the field `scores`.

        Returns:
            prediction (BoxList): the detected objects. Additional information
                of the detection properties can be found in the fields of
                the BoxList via `prediction.fields()`
        """
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels")
        thresholds = self.thre_per_classes[(labels - 1).long()]
        keep = torch.nonzero(scores > thresholds).squeeze(1)
        predictions = predictions[keep]
        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)

        return predictions[idx]

    def draw_boxes(self, image, predictions):
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        colors = labels[:, None] * self.palette
        colors = (colors % 255).numpy().astype("uint8").tolist()

        for box, color in zip(boxes, colors):
            box = box.to(torch.int64)
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist()
            image = cv2.rectangle(image, tuple(top_left), tuple(bottom_right), tuple(color), 2)

        return image

    def draw_class_names(self, image, predictions):
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        labels = [self.cfg.class_names[i] for i in labels]
        boxes = predictions.bbox

        for box, score, label in zip(boxes, scores, labels):
            x, y = box[:2]
            cv2.putText(image, f'{label}: {score:.2f}', (x, y), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)

        return image
