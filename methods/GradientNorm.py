from EarlyTrain import EarlyTrain
import torch
import numpy as np
from .. import nets


class GradientNorm(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, all_param=False,
                 specific_model=None, balance=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model

        if specific_model is not None and specific_model not in nets.model_choices:
            self.specific_model = None

        self.all_param = all_param
        self.balance = balance

    def after_loss(self, outputs, loss, predicted, targets, batch_inds, epoch):
        for index, loss_val in zip(batch_inds, loss):
            # Save gradient of parameters of the model into one tensor
            # If self.all_param==False, only calculate the gradient of the last layer,
            # Otherwise, save gradients of all parameters.
            if self.all_param:
                loss_val.backward(retain_graph=True)
                self.norm_matrix[index, epoch] = torch.norm(
                    torch.cat([torch.flatten(p.grad) for p in self.model.parameters() if p.requires_grad]), p=2)
            else:
                self.norm_matrix[index, epoch] = torch.norm(torch.cat(
                    [torch.flatten(torch.autograd.grad(loss_val, p, retain_graph=True)[0]) for p in
                     self.model.get_last_layer().parameters() if p.requires_grad]), p=2)

    def while_update(self, loss, predicted, targets, epoch, batch_idx, batch_size):
        print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def before_run(self):
        # Initialize a matrix to save norms of each sample
        self.norm_matrix = torch.zeros([self.n_train, self.epochs], requires_grad=False).to(self.args.device)

    def select(self, **kwargs):
        self.run()
        self.norm_mean = torch.mean(self.norm_matrix, dim=1).numpy()
        if not self.balance:
            top_examples = self.train_indx[np.argsort(self.norm_mean)][::-1][:self.coreset_size]
        else:
            top_examples = np.array([], dtype=np.int64)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                budget = round(self.fraction * len(c_indx))
                top_examples = np.append(top_examples, c_indx[np.argsort(self.norm_mean[c_indx])[::-1][:budget]])

        return torch.utils.data.Subset(self.dst_train, top_examples), top_examples
