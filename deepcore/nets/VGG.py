import torch.nn as nn
from torch import set_grad_enabled
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


class _VGG(nn.Module):
    def __init__(self, vgg_name, channel, num_classes, norm='instancenorm', record_embedding=False, no_grad=False):
        super(_VGG, self).__init__()
        self.channel = channel
        self.features = self._make_layers(cfg_vgg[vgg_name], norm)
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

    def _make_layers(self, cfg, norm):
        layers = []
        in_channels = self.channel
        for ic, x in enumerate(cfg):
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=3 if self.channel == 1 and ic == 0 else 1),
                           nn.GroupNorm(x, x, affine=True) if norm == 'instancenorm' else nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


def VGG(arch: str, channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
        pretrained: bool = False):
    arch = arch.lower()
    if pretrained:
        if im_size[0] != 224 or im_size[1] != 224:
            raise NotImplementedError("torchvison pretrained models only accept inputs with size of 224*224")
        try:
            net = vgg.__dict__[arch](pretrained=True)
        except:
            raise ValueError("Model architecture not found.")
        if channel != 3:
            net.features[0] = nn.Conv2d(channel, 64, kernel_size=3, padding=1)

        from .nets_utils import alexnet_forward, alexnet_get_last_layer
        net.fc = net.classifier[-1] if num_classes == 1000 else nn.Linear(4096, num_classes)
        net.embedding_recorder = EmbeddingRecorder(record_embedding)
        net.classifier[-1] = net.embedding_recorder
        net.classifier.add_module("fc", net.fc)

        net.no_grad = no_grad

        net.old_forward = net.forward
        from types import MethodType
        net.forward = MethodType(alexnet_forward, net)
        net.get_last_layer = MethodType(alexnet_get_last_layer, net)

    elif im_size[0] == 224 and im_size[1] == 224:
        # Use torchvision models without pretrained parameters
        try:
            net = vgg.__dict__[arch]()
        except:
            raise ValueError("Model architecture not found.")
        if channel != 3:
            net.features[0] = nn.Conv2d(channel, 64, kernel_size=3, padding=1)

        from .nets_utils import alexnet_forward, alexnet_get_last_layer
        net.fc = net.classifier[-1] if num_classes == 1000 else nn.Linear(4096, num_classes)
        net.embedding_recorder = EmbeddingRecorder(record_embedding)
        net.classifier[-1] = net.embedding_recorder
        net.classifier.add_module("fc", net.fc)

        net.no_grad = no_grad

        net.old_forward = net.forward
        from types import MethodType
        net.forward = MethodType(alexnet_forward, net)
        net.get_last_layer = MethodType(alexnet_get_last_layer, net)
    elif (channel == 1 and im_size[0] == 28 and im_size[1] == 28) or (
            channel == 3 and im_size[0] == 32 and im_size[1] == 32):
        if arch == "vgg11":
            net = _VGG(arch, channel, num_classes)
        elif arch == "vgg11bn":
            net = _VGG(arch, channel, num_classes, norm='batchnorm')
        elif arch == "vgg13":
            net = _VGG(arch, channel, num_classes)
        elif arch == "vgg16":
            net = _VGG(arch, channel, num_classes)
        elif arch == "vgg19":
            net = _VGG(arch, channel, num_classes)
        else:
            raise ValueError("Model architecture not found.")
    else:
        raise NotImplementedError("Network Architecture for current dataset has not been implemented.")
    return net


def VGG11(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
          pretrained: bool = False):
    return VGG("vgg11", channel, num_classes, im_size, record_embedding, no_grad, pretrained)
    # return VGG('VGG11', channel, num_classes)


def VGG11BN(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
            pretrained: bool = False):
    return VGG('vgg11bn', channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def VGG13(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
          pretrained: bool = False):
    return VGG('vgg13', channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def VGG16(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
          pretrained: bool = False):
    return VGG('vgg16', channel, num_classes, im_size, record_embedding, no_grad, pretrained)


def VGG19(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
          pretrained: bool = False):
    return VGG('vgg19', channel, num_classes, im_size, record_embedding, no_grad, pretrained)
