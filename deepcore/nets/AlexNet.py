import torch.nn as nn
from torch import set_grad_enabled
from torchvision.models import alexnet
from .nets_utils import EmbeddingRecorder


# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,

class _AlexNet(nn.Module):
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


def AlexNet(channel: int, num_classes: int, im_size, record_embedding: bool = False, no_grad: bool = False,
            pretrained: bool = False):
    if pretrained:
        if im_size[0] != 224 or im_size[1] != 224:
            raise NotImplementedError("torchvison pretrained models only accept inputs with size of 224*224")
        net = alexnet(pretrained=True)

        if channel != 3:
            net.features[0] = nn.Conv2d(channel, 64, kernel_size=11, stride=4, padding=2)

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
        net = alexnet(num_classes=num_classes)

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
        net = _AlexNet(channel, num_classes, record_embedding, no_grad)
    else:
        raise NotImplementedError("Network Architecture for current dataset has not been implemented.")
    return net


'''
from types import MethodType
net.forward=MethodType(ff, net)


class RecordEmbedding(nn.Module):
	def __init__(self, network):
		super().__init__()
		self.network=network
	def forward(self, x):
		if self.network.record_embedding:
			self.network.embedding = x
		return x

class _AlexNet(models.AlexNet):
	def __init__(self, channel, num_classes: int = 1000, record_embedding=False, no_grad=False) -> None:
		super().__init__(num_classes)
		self.record_embedding = record_embedding
		self.no_grad = no_grad
		self.fc_original=net.classifier[-1]
		self.classifier[-1]=RecordEmbedding(self)
		self.classifier.add_module("fc_original", self.fc_original)
	def forward(self, x: torch.Tensor) -> torch.Tensor:
		with set_grad_enabled(not self.no_grad):
			x = super().forward(x)
		return x
'''

'''

net = models.AlexNet()

inputs=torch.tensor(np.random.random(4*3*224*224)).view(4,3,224,224).type(torch.float)
net(inputs)
sum([p.view(-1).shape[0] for p in net.parameters() if p.requires_grad])

net.fc_original=net.classifier[-1]

class RecordEmbedding(nn.Module):
	def __init__(self, network):
		super().__init__()
		self.network=network
	def forward(self, x):
		if self.network.record_embedding:
			self.network.embedding = x
		return x


net.classifier[-1]=RecordEmbedding(net)
net.classifier.add_module("fc_original", net.fc_original)

net.no_grad = False
net.record_embedding = False


net.old_forward=net.forward
def ff(self, x: torch.Tensor) -> torch.Tensor:
	with set_grad_enabled(not self.no_grad):
		x = self.old_forward(x)
	return x


from types import MethodType
net.forward=MethodType(ff, net)
'''
