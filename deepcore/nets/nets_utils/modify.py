from torch import nn, set_grad_enabled, Tensor


class EmbeddingRecorder(nn.Module):
    def __init__(self, record_embedding: bool = False):
        super().__init__()
        self.record_embedding = record_embedding

    def forward(self, x):
        if self.record_embedding:
            self.embedding = x
        return x


def alexnet_forward(self, x: Tensor) -> Tensor:
    with set_grad_enabled(not self.no_grad):
        x = self.old_forward(x)
    return x


def alexnet_get_last_layer(self):
    return self.fc


def resnet_forward(self, x: Tensor) -> Tensor:
    with set_grad_enabled(not self.no_grad):
        x = self._forward_impl(x)
    return x


def resnet_get_last_layer(self):
    return self.real_fc
