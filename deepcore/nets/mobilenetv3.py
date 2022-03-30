import torch.nn as nn
from torch import set_grad_enabled, flatten, Tensor
from torchvision.models import mobilenetv3
from .nets_utils import EmbeddingRecorder
import math

'''MobileNetV3 in PyTorch.
Paper： "Inverted Residuals and Linear Bottlenecks:Mobile Networks for Classification, Detection and Segmentation" 

Acknowlegement to:
https://github.com/d-li14/mobilenetv3.pytorch/blob/master/mobilenetv3.py
'''


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SELayer(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, _make_divisible(channel // reduction, 8)),
            nn.ReLU(inplace=True),
            nn.Linear(_make_divisible(channel // reduction, 8), channel),
            h_sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


def conv_3x3_bn(inp, oup, stride, padding=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, padding, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        h_swish()
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, hidden_dim, oup, kernel_size, stride, use_se, use_hs):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        self.identity = stride == 1 and inp == oup

        if inp == hidden_dim:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, (kernel_size - 1) // 2, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                # Squeeze-and-Excite
                SELayer(hidden_dim) if use_se else nn.Identity(),
                h_swish() if use_hs else nn.ReLU(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3_32x32(nn.Module):
    def __init__(self, cfgs, mode, channel=3, num_classes=1000, record_embedding=False,
                 no_grad=False, width_mult=1.):
        super(MobileNetV3_32x32, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = cfgs
        assert mode in ['mobilenet_v3_large', 'mobilenet_v3_small']

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

        # building first layer
        input_channel = _make_divisible(16 * width_mult, 8)
        layers = [conv_3x3_bn(channel, input_channel, 2, padding=3 if channel == 1 else 1)]
        # building inverted residual blocks
        block = InvertedResidual
        for k, t, c, use_se, use_hs, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, 8)
            exp_size = _make_divisible(input_channel * t, 8)
            layers.append(block(input_channel, exp_size, output_channel, k, s, use_se, use_hs))
            input_channel = output_channel
        self.features = nn.Sequential(*layers)
        # building last several layers
        self.conv = conv_1x1_bn(input_channel, exp_size)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        output_channel = {'mobilenet_v3_large': 1280, 'mobilenet_v3_small': 1024}
        output_channel = _make_divisible(output_channel[mode] * width_mult, 8) if width_mult > 1.0 else output_channel[
            mode]
        self.classifier = nn.Sequential(
            nn.Linear(exp_size, output_channel),
            h_swish(),
            nn.Dropout(0.2),
            self.embedding_recorder,
            nn.Linear(output_channel, num_classes),
        )

        self._initialize_weights()

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            x = self.features(x)
            x = self.conv(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def get_last_layer(self):
        return self.classifier[-1]


class MobileNetV3_224x224(mobilenetv3.MobileNetV3):
    def __init__(self, inverted_residual_setting, last_channel,
                 channel=3, num_classes=1000, record_embedding=False, no_grad=False, **kwargs):
        super(MobileNetV3_224x224, self).__init__(inverted_residual_setting, last_channel,
                                                  num_classes=num_classes, **kwargs)

        self.embedding_recorder = EmbeddingRecorder(record_embedding)

        self.fc = self.classifier[-1]
        self.classifier[-1] = self.embedding_recorder
        self.classifier.add_module("fc", self.fc)

        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc

    def _forward_impl(self, x: Tensor) -> Tensor:
        with set_grad_enabled(not self.no_grad):
            x = self.features(x)
            x = self.avgpool(x)
            x = flatten(x, 1)
            x = self.classifier(x)
            return x


def MobileNetV3(arch: str, channel: int, num_classes: int, im_size, record_embedding: bool = False,
                no_grad: bool = False,
                pretrained: bool = False, **kwargs):
    arch = arch.lower()
    if pretrained:
        if channel != 3:
            raise NotImplementedError("Network Architecture for current dataset has not been implemented.")

        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(arch)
        net = MobileNetV3_224x224(inverted_residual_setting=inverted_residual_setting, last_channel=last_channel,
                                  channel=3, num_classes=1000, record_embedding=record_embedding, no_grad=no_grad,
                                  **kwargs)

        from torch.hub import load_state_dict_from_url
        state_dict = load_state_dict_from_url(mobilenetv3.model_urls[arch], progress=True)
        net.load_state_dict(state_dict)

        if num_classes != 1000:
            net.fc = nn.Linear(last_channel, num_classes)
            net.classifier[-1] = net.fc

    elif im_size[0] == 224 and im_size[1] == 224:
        if channel != 3:
            raise NotImplementedError("Network Architecture for current dataset has not been implemented.")
        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(arch)
        net = MobileNetV3_224x224(inverted_residual_setting=inverted_residual_setting, last_channel=last_channel,
                                  channel=channel, num_classes=num_classes, record_embedding=record_embedding,
                                  no_grad=no_grad, **kwargs)

    elif (channel == 1 and im_size[0] == 28 and im_size[1] == 28) or (
            channel == 3 and im_size[0] == 32 and im_size[1] == 32):
        if arch == "mobilenet_v3_large":
            cfgs = [
                # k, t, c, SE, HS, s
                [3, 1, 16, 0, 0, 1],
                [3, 4, 24, 0, 0, 2],
                [3, 3, 24, 0, 0, 1],
                [5, 3, 40, 1, 0, 2],
                [5, 3, 40, 1, 0, 1],
                [5, 3, 40, 1, 0, 1],
                [3, 6, 80, 0, 1, 2],
                [3, 2.5, 80, 0, 1, 1],
                [3, 2.3, 80, 0, 1, 1],
                [3, 2.3, 80, 0, 1, 1],
                [3, 6, 112, 1, 1, 1],
                [3, 6, 112, 1, 1, 1],
                [5, 6, 160, 1, 1, 2],
                [5, 6, 160, 1, 1, 1],
                [5, 6, 160, 1, 1, 1]
            ]
            net = MobileNetV3_32x32(cfgs, arch, channel=channel, num_classes=num_classes,
                                    record_embedding=record_embedding, no_grad=no_grad)
        elif arch == "mobilenet_v3_small":
            cfgs = [
                # k, t, c, SE, HS, s
                [3, 1, 16, 1, 0, 2],
                [3, 4.5, 24, 0, 0, 2],
                [3, 3.67, 24, 0, 0, 1],
                [5, 4, 40, 1, 1, 2],
                [5, 6, 40, 1, 1, 1],
                [5, 6, 40, 1, 1, 1],
                [5, 3, 48, 1, 1, 1],
                [5, 3, 48, 1, 1, 1],
                [5, 6, 96, 1, 1, 2],
                [5, 6, 96, 1, 1, 1],
                [5, 6, 96, 1, 1, 1],
            ]
            net = MobileNetV3_32x32(cfgs, arch, channel=channel, num_classes=num_classes,
                                    record_embedding=record_embedding, no_grad=no_grad)
        else:
            raise ValueError("Model architecture not found.")
    else:
        raise NotImplementedError("Network Architecture for current dataset has not been implemented.")
    return net


def MobileNetV3Large(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
                     pretrained: bool = False, **kwargs):
    return MobileNetV3("mobilenet_v3_large", channel, num_classes, im_size, record_embedding, no_grad,
                       pretrained, **kwargs)


def MobileNetV3Small(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
                     pretrained: bool = False, **kwargs):
    return MobileNetV3("mobilenet_v3_small", channel, num_classes, im_size, record_embedding, no_grad,
                       pretrained, **kwargs)
