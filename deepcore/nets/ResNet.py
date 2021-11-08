import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from .nets_utils import EmbeddingRecorder
from torchvision.models import resnet


# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(BasicBlock, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion * planes, self.expansion * planes,
                             affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, norm='instancenorm'):
        super(Bottleneck, self).__init__()
        self.norm = norm
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(planes, planes, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(self.expansion * planes, self.expansion * planes,
                                affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(self.expansion * planes, self.expansion * planes,
                             affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class _ResNet(nn.Module):
    def __init__(self, block, num_blocks, channel=3, num_classes=10, norm='instancenorm', record_embedding=False,
                 no_grad=False):
        super(_ResNet, self).__init__()
        self.in_planes = 64
        self.norm = norm

        self.conv1 = nn.Conv2d(channel, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(64, 64, affine=True) if self.norm == 'instancenorm' else nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.classifier = nn.Linear(512 * block.expansion, num_classes)

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, self.norm))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def get_last_layer(self):
        return self.classifier

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            out = F.relu(self.bn1(self.conv1(x)))
            out = self.layer1(out)
            out = self.layer2(out)
            out = self.layer3(out)
            out = self.layer4(out)
            out = F.avg_pool2d(out, 4)
            out = out.view(out.size(0), -1)
            out = self.embedding_recorder(out)
            out = self.classifier(out)
        return out


def ResNet(arch: str, channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
           pretrained: bool = False):
    arch = arch.lower()
    if pretrained:
        if im_size[0] != 224 or im_size[1] != 224:
            raise NotImplementedError("torchvison pretrained models only accept inputs with size of 224*224")
        try:
            net = resnet.__dict__[arch](pretrained=True)
        except:
            raise ValueError("Model architecture not found.")
        if channel != 3:
            net.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        net.real_fc = net.fc if num_classes != 1000 else nn.Linear(net.fc.in_features, num_classes)
        net.embedding_recorder = EmbeddingRecorder(record_embedding)
        net.fc = nn.Sequential(net.embedding_recorder, net.real_fc)

        net.no_grad = no_grad
        from types import MethodType
        from .nets_utils import resnet_forward, resnet_get_last_layer
        net.forward = MethodType(resnet_forward, net)
        net.get_last_layer = MethodType(resnet_get_last_layer, net)

    elif im_size[0] == 224 and im_size[1] == 224:
        # Use torchvision models without pretrained parameters
        try:
            net = resnet.__dict__[arch]()
        except:
            raise ValueError("Model architecture not found.")
        if channel != 3:
            net.conv1 = nn.Conv2d(channel, 64, kernel_size=7, stride=2, padding=3, bias=False)

        net.real_fc = net.fc if num_classes != 1000 else nn.Linear(net.fc.in_features, num_classes)
        net.embedding_recorder = EmbeddingRecorder(record_embedding)
        net.fc = nn.Sequential(net.embedding_recorder, net.real_fc)

        net.no_grad = no_grad
        from types import MethodType
        from .nets_utils import resnet_forward, resnet_get_last_layer
        net.forward = MethodType(resnet_forward, net)
        net.get_last_layer = MethodType(resnet_get_last_layer, net)

    elif (channel == 1 and im_size[0] == 28 and im_size[1] == 28) or (
            channel == 3 and im_size[0] == 32 and im_size[1] == 32):
        if arch == "resnet18bn":
            net = _ResNet(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes, norm='batchnorm')
        elif arch == "resnet18":
            net = _ResNet(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes)
        elif arch == "resnet34":
            net = _ResNet(BasicBlock, [3, 4, 6, 3], channel=channel, num_classes=num_classes)
        elif arch == "resnet50":
            net = _ResNet(Bottleneck, [3, 4, 6, 3], channel=channel, num_classes=num_classes)
        elif arch == "resnet101":
            net = _ResNet(Bottleneck, [3, 4, 23, 3], channel=channel, num_classes=num_classes)
        elif arch == "resnet152":
            net = _ResNet(Bottleneck, [3, 8, 36, 3], channel=channel, num_classes=num_classes)
        else:
            raise ValueError("Model architecture not found.")
    else:
        raise NotImplementedError("Network Architecture for current dataset has not been implemented.")
    return net


def ResNet18BN(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
               pretrained: bool = False):
    return ResNet("resnet18bn", channel, num_classes, im_size, record_embedding, no_grad, pretrained)
    # return ResNet(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes, norm='batchnorm')


def ResNet18(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
             pretrained: bool = False):
    return ResNet("resnet18", channel, num_classes, im_size, record_embedding, no_grad, pretrained)
    # return ResNet(BasicBlock, [2, 2, 2, 2], channel=channel, num_classes=num_classes)


def ResNet34(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
             pretrained: bool = False):
    return ResNet("resnet34", channel, num_classes, im_size, record_embedding, no_grad, pretrained)
    # return ResNet(BasicBlock, [3, 4, 6, 3], channel=channel, num_classes=num_classes)


def ResNet50(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
             pretrained: bool = False):
    return ResNet("resnet50", channel, num_classes, im_size, record_embedding, no_grad, pretrained)
    # return ResNet(Bottleneck, [3, 4, 6, 3], channel=channel, num_classes=num_classes)


def ResNet101(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
              pretrained: bool = False):
    return ResNet("resnet101", channel, num_classes, im_size, record_embedding, no_grad, pretrained)
    # return ResNet(Bottleneck, [3, 4, 23, 3], channel=channel, num_classes=num_classes)


def ResNet152(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
              pretrained: bool = False):
    return ResNet("resnet152", channel, num_classes, im_size, record_embedding, no_grad, pretrained)
    # return ResNet(Bottleneck, [3, 8, 36, 3], channel=channel, num_classes=num_classes)
