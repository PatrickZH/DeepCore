import torch.nn as nn
import torch.nn.functional as F

# Acknowledgement to
# https://github.com/kuangliu/pytorch-cifar,
# https://github.com/BIGBALLON/CIFAR-ZOO,


''' MLP '''


class MLP(nn.Module):
    def __init__(self, channel, num_classes, record_embedding=False):
        super(MLP, self).__init__()
        self.fc_1 = nn.Linear(28 * 28 * 1 if channel == 1 else 32 * 32 * 3, 128)
        self.fc_2 = nn.Linear(128, 128)
        self.fc_3 = nn.Linear(128, num_classes)

        self.record_embedding = record_embedding

    def get_last_layer(self):
        return self.fc_3

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = F.relu(self.fc_1(out))
        out = F.relu(self.fc_2(out))
        if self.record_embedding:
            self.embedding = out
        out = self.fc_3(out)
        return out
