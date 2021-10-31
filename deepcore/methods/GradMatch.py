import torch
from torch import nn
import numpy as np
from scipy.linalg import lstsq
from .CoresetMethod import CoresetMethod


# https://github.com/krishnatejakk/GradMatch

class GradMatch(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, network=None, optimizer=None, criterion=None,
                 balance=False, dst_val=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.balance = balance
        self.dst_val = dst_val

        if network is None or optimizer is None or criterion is None:
            raise ValueError("Network, criterion or optimizer is not specified.")
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion

        self.n_param = sum([p.view(-1).shape[0] for p in network.get_last_layer().parameters() if p.requires_grad])

    def OrthogonalMP_REG_torch(self, A, b, tol=1E-4, nnz=None, positive=True, lam=0, device="cuda"):
        '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
        Acknowlegement to: https://github.com/krishnatejakk/GradMatch/blob/main/GradMatch/selectionstrategies/helpers/omp_solvers.py
        Args:
          A: design matrix of size (d, n)
          b: measurement vector of length d
          tol: solver tolerance
          nnz = maximum number of nonzero coefficients (if None set to n)
          positive: only allow positive nonzero coefficients
        Returns:
           vector of length n
        '''

        with torch.no_grad():

            AT = torch.transpose(A, 0, 1)
            d, n = A.shape
            if nnz is None:
                nnz = n
            x = torch.zeros(n, device=device)
            resid = b.detach().clone()
            normb = b.norm().item()
            indices = []

            argmin = torch.tensor([-1])
            for i in range(nnz):
                if resid.norm().item() / normb < tol:
                    break
                projections = torch.matmul(AT, resid)

                if positive:
                    index = torch.argmax(projections)
                else:
                    index = torch.argmax(torch.abs(projections))

                if index in indices:
                    break

                indices.append(index)
                if len(indices) == 1:
                    A_i = A[:, index]
                    x_i = projections[index] / torch.dot(A_i, A_i).view(-1)
                    A_i = A[:, index].view(1, -1)
                else:
                    A_i = torch.cat((A_i, A[:, index].view(1, -1)), dim=0)
                    temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device=device)
                    x_i, _ = torch.lstsq(torch.matmul(A_i, b).view(-1, 1), temp)

                    if positive:

                        while min(x_i) < 0.0:

                            argmin = torch.argmin(x_i)
                            indices = indices[:argmin] + indices[argmin + 1:]
                            A_i = torch.cat((A_i[:argmin], A_i[argmin + 1:]),
                                            dim=0)
                            if argmin.item() == A_i.shape[0]:
                                break
                            temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0],
                                                                                                   device=device)
                            x_i, _ = torch.lstsq(torch.matmul(A_i, b).view(-1, 1), temp)

                if argmin.item() == A_i.shape[0]:
                    break

                resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)

            x_i = x_i.view(-1)

            for i, index in enumerate(indices):
                try:
                    x[index] += x_i[i]
                except IndexError:
                    x[index] += x_i
        return x.view(-1).cpu().numpy()

    def OrthogonalMP_REG_numpy(self, A, b, tol=1E-4, nnz=None, positive=False, lam=0):
        '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
         Acknowlegement to: https://github.com/krishnatejakk/GradMatch/blob/main/GradMatch/selectionstrategies/helpers/omp_solvers.py
        Args:
          A: design matrix of size (d, n)
          b: measurement vector of length d
          tol: solver tolerance
          nnz = maximum number of nonzero coefficients (if None set to n)
          positive: only allow positive nonzero coefficients
        Returns:
           vector of length n
        '''

        AT = A.T
        d, n = A.shape
        if nnz is None:
            nnz = n
        x = np.zeros(n)
        resid = np.copy(b)
        normb = np.linalg.norm(b)
        indices = []

        for i in range(nnz):
            if np.linalg.norm(resid) / normb < tol:
                break
            projections = AT.dot(resid)
            if positive:
                index = np.argmax(projections)
            else:
                index = np.argmax(abs(projections))
            if index in indices:
                break
            indices.append(index)
            if len(indices) == 1:
                A_i = A[:, index]
                x_i = projections[index] / A_i.T.dot(A_i)
            else:
                A_i = np.vstack([A_i, A[:, index]])
                x_i = lstsq(A_i.dot(A_i.T) + lam * np.identity(A_i.shape[0]), A_i.dot(b))[0]
                if positive:
                    while min(x_i) < 0.0:
                        argmin = np.argmin(x_i)
                        indices = indices[:argmin] + indices[argmin + 1:]
                        A_i = np.vstack([A_i[:argmin], A_i[argmin + 1:]])
                        x_i = lstsq(A_i.dot(A_i.T) + lam * np.identity(A_i.shape[0]), A_i.dot(b))[0]
            resid = b - A_i.T.dot(x_i)

        for i, index in enumerate(indices):
            try:
                x[index] += x_i[i]
            except IndexError:
                x[index] += x_i
        return x

    def calc_gradient(self, index=None, val=False):

        if val:
            batch_loader = torch.utils.data.DataLoader(
                self.dst_val if index is None else torch.utils.data.Subset(self.dst_val, index),
                batch_size=self.args.batch)
            sample_num = len(self.dst_val.targets) if index is None else len(index)
        else:
            batch_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args.batch)
            sample_num = self.n_train if index is None else len(index)

        gradients = torch.zeros([sample_num, self.n_param], requires_grad=False).to(self.args.device)

        i = 0
        j = 0
        for input, targets in batch_loader:
            self.optimizer.zero_grad()
            outputs = torch.nn.functional.softmax(self.network(input.to(self.args.device)), dim=1)
            loss = self.criterion(outputs, targets.to(self.args.device))

            for loss_val in loss:
                gradients[j:, ] = torch.cat(
                    [torch.flatten(torch.autograd.grad(loss_val, p, retain_graph=True)[0]) for p in
                     self.network.get_last_layer().parameters() if p.requires_grad])
                j = j + 1

            i = i + 1
        return gradients

    def select(self, **kwargs):

        if self.dst_val is not None:
            val_num = len(self.dst_val.targets)

        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            weights = np.array([], dtype=np.float32)
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                cur_gradients = self.calc_gradient(class_index)
                if self.dst_val is not None:
                    # Also calculate gradients of the validation set.
                    val_class_index = np.arange(val_num)[self.dst_val.targets == c]
                    cur_val_gradients = torch.mean(self.calc_gradient(val_class_index, val=True), dim=0)
                else:
                    cur_val_gradients = torch.mean(cur_gradients, dim=0)
                if self.args.device == "cpu":
                    # Compute OMP on numpy
                    cur_weights = self.OrthogonalMP_REG_numpy(cur_gradients.numpy().T, cur_val_gradients.numpy(),
                                                              nnz=round(len(class_index) * self.fraction))
                else:
                    cur_weights = self.OrthogonalMP_REG_torch(cur_gradients.T, cur_val_gradients,
                                                              nnz=round(len(class_index) * self.fraction))
                selection_result = np.append(selection_result, class_index[np.nonzero(cur_weights)[0]])
                weights = np.append(weights, cur_weights[np.nonzero(cur_weights)[0]])
        else:
            cur_gradients = self.calc_gradient()
            if self.dst_val is not None:
                # Also calculate gradients of the validation set.
                cur_val_gradients = torch.mean(self.calc_gradient(val=True), dim=0)
            else:
                cur_val_gradients = torch.mean(cur_gradients, dim=0)
            if self.args.device == "cpu":
                # Compute OMP on numpy
                cur_weights = self.OrthogonalMP_REG_numpy(cur_gradients.numpy().T, cur_val_gradients.numpy(),
                                                          nnz=self.coreset_size)
            else:
                cur_weights = self.OrthogonalMP_REG_torch(cur_gradients.T, cur_val_gradients,
                                                          nnz=self.coreset_size)
            selection_result = np.nonzero(cur_weights)[0]
            weights = cur_weights[selection_result]
        return torch.utils.data.Subset(self.dst_train, selection_result), selection_result, weights
