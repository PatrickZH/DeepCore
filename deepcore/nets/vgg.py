import torch.nn as nn
from torch import set_grad_enabled, flatten, Tensor
from .nets_utils import EmbeddingRecorder
from torchvision.models import vgg

# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

cfg_vgg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG_32x32(nn.Module):
    def __init__(self, vgg_name, channel, num_classes, record_embedding=False, no_grad=False):
        super(VGG_32x32, self).__init__()
        self.channel = channel
        self.features = self._make_layers(cfg_vgg[vgg_name])
        self.classifier = nn.Linear(512 if vgg_name != 'VGGS' else 128, num_classes)

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.embedding_recorder(x)
            x = self.classifier(x)
        return x

    def get_last_layer(self):
        return self.classifier

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if self.channel == 1 and ic == 0 else 1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class VGG_224x224(vgg.VGG):
    def __init__(self, features: nn.Module, channel: int, num_classes: int, record_embedding: bool = False,
                 no_grad: bool = False, **kwargs):
        super(VGG_224x224, self).__init__(features, num_classes, **kwargs)
        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        if channel != 3:
            self.features[0] = nn.Conv2d(channel, 64, kernel_size=3, padding=1)
        self.fc = self.classifier[-1]
        self.classifier[-1] = self.embedding_recorder
        self.classifier.add_module("fc", self.fc)

        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc

    def forward(self, x: Tensor) -> Tensor:
        with set_grad_enabled(not self.no_grad):
            x = self.features(x)
            x = self.avgpool(x)
            x = flatten(x, 1)
            x = self.classifier(x)
            return x


def VGG(arch: str, channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
        pretrained: bool = False):
    arch = arch.lower()
    if pretrained:
        if im_size[0] != 224 or im_size[1] != 224:
            raise NotImplementedError("torchvison pretrained models only accept inputs with size of 224*224")
        net = VGG_224x224(features=vgg.make_layers(cfg_vgg[arch], True), channel=3, num_classes=1000,
                          record_embedding=record_embedding, no_grad=no_grad)

        from torch.hub import load_state_dict_from_url
        state_dict = load_state_dict_from_url(vgg.model_urls[arch], progress=True)
        net.load_state_dict(state_dict)

        if channel != 3:
            net.features[0] = nn.Conv2d(channel, 64, kernel_size=3, padding=1)

        if num_classes != 1000:
            net.fc = nn.Linear(4096, num_classes)
            net.classifier[-1] = net.fc

    elif im_size[0] == 224 and im_size[1] == 224:
        net = VGG_224x224(features=vgg.make_layers(cfg_vgg[arch], True), channel=channel, num_classes=num_classes,
                          record_embedding=record_embedding, no_grad=no_grad)

    elif (channel == 1 and im_size[0] == 28 and im_size[1] == 28) or (
            channel == 3 and im_size[0] == 32 and im_size[1] == 32):
        net = VGG_32x32(arch, channel, num_classes=num_classes, record_embedding=record_embedding, no_grad=no_grad)
    else:
        raise NotImplementedError("Network Architecture for current dataset has not been implemented.")
    return net


def VGG11(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
          pretrained: bool = False):
    return VGG("vgg11", channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def VGG13(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
          pretrained: bool = False):
    return VGG('vgg13', channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def VGG16(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
          pretrained: bool = False):
    return VGG('vgg16', channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def VGG19(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
          pretrained: bool = False):
    return VGG('vgg19', channel, num_classes, im_size, record_embedding, no_grad, pretrained)
