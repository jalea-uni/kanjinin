import torch.nn as nn
import torchvision.models as models


def get_model(num_classes, pretrained=True):
    model = models.resnet18(pretrained=pretrained)
    # first conv expects 3-ch, replace for 1-ch
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model