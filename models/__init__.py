from .lenet import LeNet
from torchvision import models as torch_models
from torch import nn
import re

__all__ = [
    "LeNet",
]

model_list = {
    "LeNet": LeNet,
}

def get_model_class(name):
    name = re.sub("resnet", "ResNet", name)

    return name

def get_classification_model(name, num_classes, label_type, pretrain=None):
    if hasattr(torch_models, name):
        if label_type == "onehot":
            if getattr(getattr(torch_models, f"{get_model_class(name)}_Weights"), pretrain, None) is not None:
                print(f"Use pretrain: {pretrain}")
            model = getattr(torch_models, name)(weights=getattr(getattr(torch_models, f"{get_model_class(name)}_Weights"), pretrain, None))
            model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name in model_list:
        model = model_list[name](num_classes=num_classes)
    else:
        raise ValueError(f"Model {name} not found.")
        
    return model