from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def get_detection_model(num_classes=2):
    """
    Faster RCNN Object detection.
    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def get_transform(train=False):
    transforms_list = []
    if train:
        transforms.append(transforms.RandomHorizontalFlip(0.5))
    transforms_list.append(transforms.ToTensor())

    return transforms.Compose(transforms_list)
