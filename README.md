# DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning [PDF](https://arxiv.org/pdf/2204.08499.pdf)


### Introduction
To advance the research of coreset selection in deep learning, we contribute a code library named **DeepCore**, an extensive and extendable code library, for coreset selection in deep learning, reproducing dozens of popular and advanced coreset selection methods and enabling a fair comparison of different methods in the same experimental settings. **DeepCore** is highly modular, allowing to add new architectures, datasets, methods and learning scenarios easily. It is built on PyTorch.   

### Coreset Methods
We list the methods in DeepCore according to the categories in our original paper, they are 1) geometry based methods Contextual Diversity (CD), Herding  and k-Center Greedy; 2) uncertainty scores; 3) error based methods Forgetting  and GraNd score ; 4) decision boundary based methods Cal  and DeepFool ; 5) gradient matching based methods Craig  and GradMatch ; 6) bilevel optimiza- tion methods Glister ; and 7) Submodularity based Methods (GC) and Facility Location (FL) functions. we also have Random selection as the baseline.

### Datasets
It contains a series of other popular computer vision datasets, namely MNIST, QMNIST, FashionMNIST, SVHN, CIFAR10, CIFAR100 and TinyImageNet and ImageNet.

### Models
They are two-layer fully connected MLP, LeNet , AlexNet, VGG, Inception-v3, ResNet, WideResNet and MobileNet-v3.

### Example
Selecting with Glister and training on the coreset with fraction 0.1.
```sh
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset CIFAR10 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection Glister --model InceptionV3 --lr 0.1 -sp ./result --batch 128
```

Resuming interuppted training with argument ```--resume```.
```sh
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset CIFAR10 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection Glister --model InceptionV3 --lr 0.1 -sp ./result --batch 128 --resume "CIFAR10_InceptionV3_Glister_exp0_epoch200_2022-02-05 21:31:53.762903_0.1_unknown.ckpt"
```

Batch size can be seperatedly assigned for both selection and training.
```sh
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.5 --dataset ImageNet --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection Cal --model MobileNetV3Large --lr 0.1 -sp ./result -tb 256 -sb 128
```

Argument ```--uncertainty``` to choose uncertainty scores.
```sh
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset CIFAR10 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection Uncertainty --model ResNet18 --lr 0.1 -sp ./result --batch 128 --uncertainty Entropy
```


Argument ```--submodular``` to choose uncertainty scores.
```sh
CUDA_VISIBLE_DEVICES=0 python -u main.py --fraction 0.1 --dataset CIFAR10 --data_path ~/datasets --num_exp 5 --workers 10 --optimizer SGD -se 10 --selection Submodular --model ResNet18 --lr 0.1 -sp ./result --batch 128 --submodular GraphCut
```

### Extend

DeepCore is highly modular and scalable. It allows to add new architectures, datasets and selection methods easily, to help coreset methods to be evaluated in a richer set of scenarios, and also to facilitate new methods for comparison. Here is an example for datasets. To add a new dataset, you need implement a function whose input is the data path and outputs are number of channels, size of image, number of classes, names of classes, mean, std and training and testing dataset inherited from ```torch.utils.data.Dataset```.


```python
from torchvision import datasets, transforms


def MNIST(data_path):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
```
This is an example for implementing network architecture.
```python
import torch.nn as nn
import torch.nn.functional as F
from torch import set_grad_enabled
from .nets_utils import EmbeddingRecorder


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
```

To implement the new coreset method, you need to inherit the new method from the ```CoresetMethod``` class and return the selected indices via the ```select``` method.

```python
class CoresetMethod(object):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, **kwargs):
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("Illegal Coreset Size.")
        self.dst_train = dst_train
        self.num_classes = len(dst_train.classes)
        self.fraction = fraction
        self.random_seed = random_seed
        self.index = []
        self.args = args

        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)

    def select(self, **kwargs):
        return
```

### References

1. Agarwal, S., Arora, H., Anand, S., Arora, C.: Contextual diversity for active learning. In: ECCV. pp. 137–153. Springer (2020)
2. Coleman, C., Yeh, C., Mussmann, S., Mirzasoleiman, B., Bailis, P., Liang, P., Leskovec, J., Zaharia, M.: Selection via proxy: Efficient data selection for deep learning. In: ICLR (2019)
3. Ducoffe, M., Precioso, F.: Adversarial active learning for deep networks: a margin based approach. arXiv preprint arXiv:1802.09841 (2018)
4. Iyer, R., Khargoankar, N., Bilmes, J., Asanani, H.: Submodular combinatorial information measures with applications in machine learning. In: Algorithmic Learning Theory. pp. 722–754. PMLR (2021)
5. Killamsetty, K., Durga, S., Ramakrishnan, G., De, A., Iyer, R.: Grad-match: Gradient matching based data subset selection for efficient deep model training. In: ICML. pp. 5464–5474 (2021)
6. Killamsetty, K., Sivasubramanian, D., Ramakrishnan, G., Iyer, R.: Glister: Generalization based data subset selection for efficient and robust learning. In: Proceedings of the AAAI Conference on Artificial Intelligence (2021)
7. Margatina, K., Vernikos, G., Barrault, L., Aletras, N.: Active learning by acquiring contrastive examples. arXiv preprint arXiv:2109.03764 (2021)
8. Mirzasoleiman, B., Bilmes, J., Leskovec, J.: Coresets for data-efficient training of machine learning models. In: ICML. PMLR (2020)
9. Paul, M., Ganguli, S., Dziugaite, G.K.: Deep learning on a data diet: Finding important examples early in training. arXiv preprint arXiv:2107.07075 (2021)
10. Sener, O., Savarese, S.: Active learning for convolutional neural networks: A coreset approach. In: ICLR (2018)
11. Toneva, M., Sordoni, A., des Combes, R.T., Trischler, A., Bengio, Y., Gordon, G.J.: An empirical study of example forgetting during deep neural network learning. In: ICLR (2018)
12. Welling, M.: Herding dynamical weights to learn. In: Proceedings of the 26th Annual International Conference on Machine Learning. pp. 1121–1128 (2009)


