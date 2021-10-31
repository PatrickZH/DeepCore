from .CoresetMethod import CoresetMethod
import torch
from torch import nn
import numpy as np
from torchvision import models, transforms
from copy import deepcopy
from .methods_utils import euclidean_dist


# 需要加balance参数

class herding(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, metric="euclidean", embedding_model=None,
                 **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)

        if metric == "euclidean":
            self.metric = euclidean_dist
        else:
            self.metric = metric

        with torch.no_grad():
            torch.manual_seed(self.random_seed)
            np.random.seed(self.random_seed)

            if embedding_model is not None:
                if embedding_model in ["ResNet18", "ResNet34", "ResNet50", "ResNet101", "ResNet152"]:
                    model = models.__dict__[embedding_model.lower()](pretrained=True).to(args.device)
                    if args.channel != 3:
                        model.conv1 = nn.Conv2d(args.channel, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3),
                                                bias=False).to(args.device)
                elif embedding_model in ["VGG11", "VGG13", "VGG16", "VGG19"]:
                    model = models.__dict__[embedding_model.lower()](pretrained=True).to(args.device)
                    if args.channel != 3:
                        model.features[0] = nn.Conv2d(args.channel, 64, kernel_size=(3, 3), stride=(1, 1),
                                                      padding=(1, 1)).to(args.device)
                else:
                    raise NotImplementedError("%s has not be implemented." % embedding_model)

                # To use pretrained models provided by torchvision, samples will be resized to 224x224.
                if args.im_size[0] != 224 or args.im_size[1] != 224:
                    temp_dst_train = deepcopy(dst_train)
                    temp_dst_train.transform = transforms.Compose([transforms.Resize(224), temp_dst_train.transform])
                else:
                    temp_dst_train = dst_train
                self.emb_dim = 1000
            else:
                temp_dst_train = dst_train
                self.emb_dim = args.channel * args.im_size[0] * args.im_size[1]
                model = lambda x: torch.flatten(x, start_dim=1)

            self.matrix = torch.zeros([self.n_train, self.emb_dim], requires_grad=False).to(args.device)

            data_loader = torch.utils.data.DataLoader(temp_dst_train, batch_size=args.batch)

            i = 0
            for inputs, _ in data_loader:
                self.matrix[i * args.batch:min((i + 1) * args.batch, self.n_train)] = model(inputs.to(args.device))
                i = i + 1

    def select(self, **kwargs):
        with torch.no_grad():
            indices = np.arange(self.n_train)
            mu = torch.mean(self.matrix, dim=0)
            select_result = np.zeros(self.n_train, dtype=bool)

            for i in range(self.coreset_size):
                dist = self.metric(((i + 1) * mu - torch.sum(self.matrix[select_result], dim=0)).view(1, -1),
                                   self.matrix[~select_result])
                p = torch.argmax(dist).item()
                p = indices[~select_result][p]
                select_result[p] = True

        return torch.utils.data.Subset(self.dst_train, indices[select_result]), indices[select_result]
