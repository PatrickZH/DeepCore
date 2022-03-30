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

Resuming interuppted training with argument ```--resume```
```sh

```



### Extend
