from torchvision import datasets, transforms


def FashionMNIST(data_path):
    channel = 1
    im_size = (28, 28)
    num_classes = 10
    mean = [0.2861]
    std = [0.3530]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.FashionMNIST(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.FashionMNIST(data_path, train=False, download=True, transform=transform)
    class_names = dst_train.classes
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
