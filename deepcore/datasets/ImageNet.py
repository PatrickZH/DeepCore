from torchvision import datasets, transforms


def ImageNet(data_path):
    channel = 3
    im_size = (448, 448)
    num_classes = 1000
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.ImageNet(data_path, train=True, download=True, transform=transform)
    dst_test = datasets.ImageNet(data_path, train=False, download=True, transform=transform)
    class_names = dst_train.classes
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test
