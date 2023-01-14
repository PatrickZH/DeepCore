import torch
import numpy as np
from scipy.linalg import lstsq
from scipy.optimize import nnls
from .earlytrain import EarlyTrain
from ..nets.nets_utils import MyDataParallel


# https://github.com/krishnatejakk/GradMatch

class GradMatch(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None,
                 balance=True, dst_val=None, lam: float = 1., **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)
        self.balance = balance
        self.dst_val = dst_val

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def orthogonal_matching_pursuit(self, A, b, budget: int, lam: float = 1.):
        '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
        Acknowlegement to:
        https://github.com/krishnatejakk/GradMatch/blob/main/GradMatch/selectionstrategies/helpers/omp_solvers.py
        Args:
          A: design matrix of size (d, n)
          b: measurement vector of length d
          budget: selection budget
          lam: regularization coef. for the final output vector
        Returns:
           vector of length n
        '''
        with torch.no_grad():
            d, n = A.shape
            if budget <= 0:
                budget = 0
            elif budget > n:
                budget = n

            x = np.zeros(n, dtype=np.float32)
            resid = b.clone()
            indices = []
            boolean_mask = torch.ones(n, dtype=bool, device="cuda")
            all_idx = torch.arange(n, device='cuda')

            for i in range(budget):
                if i % self.args.print_freq == 0:
                    print("| Selecting [%3d/%3d]" % (i + 1, budget))
                projections = torch.matmul(A.T, resid)
                index = torch.argmax(projections[boolean_mask])
                index = all_idx[boolean_mask][index]

                indices.append(index.item())
                boolean_mask[index] = False

                if indices.__len__() == 1:
                    A_i = A[:, index]
                    x_i = projections[index] / torch.dot(A_i, A_i).view(-1)
                    A_i = A[:, index].view(1, -1)
                else:
                    A_i = torch.cat((A_i, A[:, index].view(1, -1)), dim=0)
                    temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device="cuda")
                    x_i, _ = torch.lstsq(torch.matmul(A_i, b).view(-1, 1), temp)
                resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)
            if budget > 1:
                x_i = nnls(temp.cpu().numpy(), torch.matmul(A_i, b).view(-1).cpu().numpy())[0]
                x[indices] = x_i
            elif budget == 1:
                x[indices[0]] = 1.
        return x

    def orthogonal_matching_pursuit_np(self, A, b, budget: int, lam: float = 1.):
        '''approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
        Acknowlegement to:
        https://github.com/krishnatejakk/GradMatch/blob/main/GradMatch/selectionstrategies/helpers/omp_solvers.py
        Args:
          A: design matrix of size (d, n)
          b: measurement vector of length d
          budget: selection budget
          lam: regularization coef. for the final output vector
        Returns:
           vector of length n
        '''
        d, n = A.shape
        if budget <= 0:
            budget = 0
        elif budget > n:
            budget = n

        x = np.zeros(n, dtype=np.float32)
        resid = np.copy(b)
        indices = []
        boolean_mask = np.ones(n, dtype=bool)
        all_idx = np.arange(n)

        for i in range(budget):
            if i % self.args.print_freq == 0:
                print("| Selecting [%3d/%3d]" % (i + 1, budget))
            projections = A.T.dot(resid)
            index = np.argmax(projections[boolean_mask])
            index = all_idx[boolean_mask][index]

            indices.append(index.item())
            boolean_mask[index] = False

            if indices.__len__() == 1:
                A_i = A[:, index]
                x_i = projections[index] / A_i.T.dot(A_i)
            else:
                A_i = np.vstack([A_i, A[:, index]])
                x_i = lstsq(A_i.dot(A_i.T) + lam * np.identity(A_i.shape[0]), A_i.dot(b))[0]
            resid = b - A_i.T.dot(x_i)
        if budget > 1:
            x_i = nnls(A_i.dot(A_i.T) + lam * np.identity(A_i.shape[0]), A_i.dot(b))[0]
            x[indices] = x_i
        elif budget == 1:
            x[indices[0]] = 1.
        return x

    def calc_gradient(self, index=None, val=False):
        self.model.eval()
        if val:
            batch_loader = torch.utils.data.DataLoader(
                self.dst_val if index is None else torch.utils.data.Subset(self.dst_val, index),
                batch_size=self.args.selection_batch, num_workers=self.args.workers)
            sample_num = len(self.dst_val.targets) if index is None else len(index)
        else:
            batch_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args.selection_batch, num_workers=self.args.workers)
            sample_num = self.n_train if index is None else len(index)

        self.embedding_dim = self.model.get_last_layer().in_features
        gradients = torch.zeros([sample_num, self.args.num_classes * (self.embedding_dim + 1)],
                                requires_grad=False, device=self.args.device)

        for i, (input, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device)).requires_grad_(True)
            loss = self.criterion(outputs, targets.to(self.args.device)).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs, retain_graph=True)[0].cpu()
                weight_parameters_grads = self.model.embedding_recorder.embedding.cpu().view(batch_num, 1,
                                                    self.embedding_dim).repeat(1,self.args.num_classes,1) *\
                                                    bias_parameters_grads.view(batch_num, self.args.num_classes,
                                                    1).repeat(1, 1, self.embedding_dim)
                gradients[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] =\
                    torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1)

        return gradients

    def finish_run(self):
        if isinstance(self.model, MyDataParallel):
            self.model = self.model.module

        self.model.no_grad = True
        with self.model.embedding_recorder:
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
                        cur_weights = self.orthogonal_matching_pursuit_np(cur_gradients.numpy().T,
                                                                          cur_val_gradients.numpy(),
                                                                        budget=round(len(class_index) * self.fraction))
                    else:
                        cur_weights = self.orthogonal_matching_pursuit(cur_gradients.to(self.args.device).T,
                                                                       cur_val_gradients.to(self.args.device),
                                                                       budget=round(len(class_index) * self.fraction))
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
                    cur_weights = self.orthogonal_matching_pursuit_np(cur_gradients.numpy().T,
                                                                      cur_val_gradients.numpy(),
                                                                      budget=self.coreset_size)
                else:
                    cur_weights = self.orthogonal_matching_pursuit(cur_gradients.T, cur_val_gradients,
                                                                   budget=self.coreset_size)
                selection_result = np.nonzero(cur_weights)[0]
                weights = cur_weights[selection_result]
        self.model.no_grad = False
        return {"indices": selection_result, "weights": weights}

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result

