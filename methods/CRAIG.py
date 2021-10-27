from CoresetMethod import CoresetMethod
import torch
from apricot.functions.facilityLocation import FacilityLocationSelection
import numpy as np


class CRAIG(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, network=None, optimizer=None, criterion=None,
                 balance=True, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.n_train = len(dst_train.targets)
        self.coreset_size = round(self.n_train * fraction)
        if network is None or optimizer is None or criterion is None:
            raise ValueError("Network, criterion or optimizer is not specified.")
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion
        self.balance = balance
        self.n_param = sum([p.view(-1).shape[0] for p in network.get_last_layer().parameters() if p.requires_grad])

    def euclidean_dist(self, x, y):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(1, -2, x, y.t())
        dist = dist.clamp(min=1e-12).sqrt()
        return dist

    def euclidean_dist(self, x):
        m = x.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, m)
        dist = xx + xx.t()
        dist.addmm_(1, -2, x, x.t())
        dist = dist.clamp(min=1e-12).sqrt()
        return dist

    def calc_gradient(self, index):
        sample_num = len(index)
        batch_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.dst_train, index),
                                                   batch_size=self.args.batch)

        gradients = torch.zeros([sample_num, self.n_param]).to(self.args.device)

        i = 0
        j = 0
        for input, targets in batch_loader:
            self.optimizer.zero_grad()
            outputs = self.network(input.to(self.args.device))
            loss = self.criterion(outputs, targets.to(self.args.device))

            for loss_val in loss:
                gradients[j:, ] = torch.cat(
                    [torch.flatten(torch.autograd.grad(loss_val, p, retain_graph=True)[0]) for p in
                     self.network.get_last_layer().parameters() if p.requires_grad])
                j = j + 1

            i = i + 1
        return self.euclidean_dist(gradients)

    def calc_weights(self, matrix, result):
        min_sample = torch.argmin(matrix[result], dim=0)
        weights = np.zeros(len(result))
        for i in min_sample:
            weights[i] = weights[i] + 1
        return weights

    def select(self, **kwargs):

        if self.balance:
            # Do selection by class
            selection_result = []
            weights = np.array([])
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                matrix = self.calc_gradient(class_index)
                class_result = FacilityLocationSelection(random_state=self.random_seed, metric='precomputed',
                                                         n_samples=round(len(class_index) * self.fraction),
                                                         optimizer="lazy").fit_transform(matrix)
                selection_result.append(class_index[class_result])
                weights = np.append(weights, self.calc_weights(matrix, class_result))
        else:
            matrix = torch.zeros([self.n_train, self.n_train])
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                matrix[class_index, class_index] = self.calc_gradient(class_index)
            selection_result = FacilityLocationSelection(random_state=self.random_seed, metric='precomputed',
                                                         n_samples=round(len(class_index) * self.fraction),
                                                         optimizer="lazy").fit_transform(matrix)
            weights = self.calc_weights(matrix, selection_result)
        return torch.utils.data.Subset(self.dst_train, selection_result), selection_result, weights
