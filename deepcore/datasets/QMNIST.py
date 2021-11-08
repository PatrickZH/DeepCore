from torchvision import datasets, transforms


def QMNIST(data_path):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.1308]
    std = [0.3088]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.QMNIST(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.QMNIST(data_path, train=False, download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]
    dst_train.targets = dst_train.targets[:, 0]
    dst_test.targets = dst_test.targets[:, 0]
    dst_train.compat = False
    dst_test.compat = False
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
