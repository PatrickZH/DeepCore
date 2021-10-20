from torchvision import datasets, transforms

def SVHN(data_path):
    channel = 3
    im_size = (32, 32)
    num_classes = 10
    mean = [0.4377, 0.4438, 0.4728]
    std = [0.1980, 0.2010, 0.1970]
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    dst_train = datasets.SVHN(data_path, split='train', download=True, transform=transform)  # no augmentation
    dst_test = datasets.SVHN(data_path, split='test', download=True, transform=transform)
    class_names = [str(c) for c in range(num_classes)]
    return channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test