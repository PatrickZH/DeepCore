import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from .nets_utils import EmbeddingRecorder

# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,


''' MLP '''


class MLP(nn.Module):
    def __init__(self, channel, num_classes, im_size, record_embedding: bool = False, no_grad: bool = False,
                 pretrained: bool = False):
        if pretrained:
            raise NotImplementedError("torchvison pretrained models not available.")
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(im_size[0] * im_size[1] * channel, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

        self.embedding_recorder = EmbeddingRecorder(record_embedding)
        self.no_grad = no_grad

    def get_last_layer(self):
        return self.fc_3

    def forward(self, x):
        with set_grad_enabled(not self.no_grad):
            out = x.view(x.size(0), -1)
            out = F.relu(self.fc_1(out))
            out = F.relu(self.fc_2(out))
            out = self.embedding_recorder(out)
            out = self.fc_3(out)
        return out
