import torch.nn as nn
from torch import set_grad_enabled
from torchvision import models
import torch
from .nets_utils import EmbeddingRecorder


# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

class AlexNet_32x32(nn.Module):
    def __init__(self, channel, num_classes, record_embedding=False, no_grad=False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(channel, 128, kernel_size=5, stride=1, padding=4 if channel == 1 else 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc = nn.Linear(192 * 4 * 4, num_classes)

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.embedding_recorder(x)
            x = self.fc(x)
        return x


class AlexNet_224x224(models.AlexNet):
    def __init__(self, channel: int, num_classes: int, record_embedding: bool = False,
                 no_grad: bool = False, **kwargs):
        super().__init__(num_classes, **kwargs)
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        if channel != 3:
            self.features[0] = nn.Conv2d(channel, 64, kernel_size=11, stride=4, padding=2)
        self.fc = self.classifier[-1]
        self.classifier[-1] = self.embedding_recorder
        self.classifier.add_module("fc", self.fc)

        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with set_grad_enabled(not self.no_grad):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        return x


def AlexNet(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
            pretrained: bool = False):
    if pretrained:
        if im_size[0] != 224 or im_size[1] != 224:
            raise NotImplementedError("torchvison pretrained models only accept inputs with size of 224*224")
        net = AlexNet_224x224(channel=3, num_classes=1000, record_embedding=record_embedding, no_grad=no_grad)

        from torch.hub import load_state_dict_from_url
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/alexnet-owt-7be5be79.pth'
                                              , progress=True)
        net.load_state_dict(state_dict)

        if channel != 3:
            net.features[0] = nn.Conv2d(channel, 64, kernel_size=11, stride=4, padding=2)
        if num_classes != 1000:
            net.fc = nn.Linear(4096, num_classes)
            net.classifier[-1] = net.fc

    elif im_size[0] == 224 and im_size[1] == 224:
        net = AlexNet_224x224(channel=channel, num_classes=num_classes, record_embedding=record_embedding,
                              no_grad=no_grad)

    elif (channel == 1 and im_size[0] == 28 and im_size[1] == 28) or (
            channel == 3 and im_size[0] == 32 and im_size[1] == 32):
        net = AlexNet_32x32(channel=channel, num_classes=num_classes, record_embedding=record_embedding,
                            no_grad=no_grad)
    else:
        raise NotImplementedError("Network Architecture for current dataset has not been implemented.")
    return net
