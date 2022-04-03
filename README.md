# DeepCore: A Comprehensive Library for Coreset Selection in Deep Learning


### Introduction
To advance the research of coreset selection in deep learning, we contribute a code library named **DeepCore**, an extensive and extendable code library, for coreset selection in deep learning, reproducing dozens of popular and advanced coreset selection methods and enabling a fair comparison of different methods in the same experimental settings. **DeepCore** is highly modular, allowing to add new architectures, datasets, methods and learning scenarios easily. It is built on PyTorch.   

### Coreset Methods
We list the methods in DeepCore according to the categories in our original paper, they are 1) geometry based methods Contextual Diversity (CD), Herding  and k-Center Greedy; 2) uncertainty scores; 3) error based methods Forgetting  and GraNd score ; 4) decision boundary based methods Cal  and DeepFool ; 5) gradient matching based methods Craig  and GradMatch ; 6) bilevel optimiza- tion methods Glister ; and 7) Submodularity based Methods (GC) and Facility Location (FL) functions. we also have Random selection as the baseline.

### Datasets
It contains a series of other popular computer vision datasets, namely MNIST, QMNIST [54], FashionMNIST, SVHN, CIFAR10, CIFAR100 and TinyImageNet and ImageNet.

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

DeepCore is highly modular and scalable. It allows to add new architectures, datasets and selection methods easily, to help coreset methods to be evaluated in a richer set of scenarios, and also to facilitate new methods for comparison. Here is an example for datasets. To add a new dataset, you need implement a function whose input is the data path and outputs are  


```python
from torchvision import datasets, transforms
import numpy as np


def MNIST(data_path, permuted=False, permutation_seed=None):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1307]
    std = [0.3081]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    if permuted:
        np.random.seed(permutation_seed)
        pixel_permutation = np.random.permutation(28 * 28)
        transform = transforms.Compose(
            [transform, transforms.Lambda(lambda x: x.view(-1, 1)[pixel_permutation].view(1, 28, 28))])

    dst_train = datasets.MNIST(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.MNIST(data_path, train=False, download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
```







