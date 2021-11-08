from .EarlyTrain import EarlyTrain
from .kCenterGreedy import kCenterGreedy
import torch


# Acknowlegement to:
# https://github.com/sharat29ag/CDAL

class ContextualDiversity(EarlyTrain, kCenterGreedy):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, already_selected=[], epochs=200,
                 specific_model=None, balance=True, **kwargs):
        EarlyTrain.__init__(self, dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)
        self.already_selected = already_selected
        if self.already_selected.__len__() != 0:
            if min(already_selected) < 0 or max(already_selected) >= self.n_train:
                raise ValueError("List of already selected points out of the boundary.")

        self.balance = balance

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def finish_run(self):
        kCenterGreedy.__init__(self, self.dst_train, self.args, self.fraction, self.random_seed, self.already_selected, self._metric, self.model)
        #
        # batch_loader = torch.utils.data.DataLoader(self.dst_train, batch_size=self.args.selection_batch)
        # for i, (inputs, _) in enumerate(batch_loader):
        #     self.matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, self.n_train)] = torch.nn.functional.softmax(self.model(inputs.to(self.args.device)), dim=1)

    def _metric(self, a_output, b_output):
        with torch.no_grad():
            # Overload self.metric function for kCenterGreedy Algorithm
            aa = a_output.view(a_output.shape[0], 1, a_output.shape[1]).repeat(1, b_output.shape[0], 1)
            bb = b_output.view(1, b_output.shape[0], b_output.shape[1]).repeat(a_output.shape[0], 1, 1)
            return torch.sum(0.5 * aa * torch.log(aa / bb) + 0.5 * bb * torch.log(bb / aa), dim=2)

    def select(self, **kwargs):
        self.metric = self._metric
        self.run()
        return kCenterGreedy.select(self)
'''
class ContextualDiversity(EarlyTrain, kCenterGreedy):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, already_selected=[], epochs=200,
                 specific_model=None, balance=True, **kwargs):
        EarlyTrain.__init__(self, dst_train, args, fraction, random_seed, epochs, specific_model)
        self.already_selected = already_selected
        if self.already_selected.__len__() != 0:
            if min(already_selected) < 0 or max(already_selected) >= self.n_train:
                raise ValueError("List of already selected points out of the boundary.")

        self.balance = balance

        # Matrix for kCenterGreedy Algorithm
        self.matrix = torch.arange(self.n_train, requires_grad=False)

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx%self.args.print_freq==0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def _metric(self, a, b):
        with torch.no_grad():
            # Overload self.metric function for kCenterGreedy Algorithm
            dis_matrix = torch.ones([a.shape[0], b.shape[0]], requires_grad=False).to(self.args.device)

            a_output = torch.nn.functional.softmax(self.model(
                next(iter(torch.utils.data.DataLoader(torch.utils.data.Subset(self.dst_train, a), batch_size=a.shape[0])))[
                    0].to(self.args.device)), dim=1)
            aa = a_output.view(a_output.shape[0], 1, a_output.shape[1]).repeat(1, self.args.selection_batch, 1)
            b_loader = torch.utils.data.DataLoader(torch.utils.data.Subset(self.dst_train, b), batch_size=self.args.selection_batch)
            for i, (inputs, _) in enumerate(b_loader):
                b_output = torch.nn.functional.softmax(self.model(inputs.to(self.args.device)), dim=1)
                bb = b_output.view(1, b_output.shape[0], b_output.shape[1]).repeat(a_output.shape[0], 1, 1)
                try:
                    dis_matrix[:a_output.shape[0], i * self.args.selection_batch:((i + 1) * self.args.selection_batch)] = torch.sum(0.5 * aa * torch.log(aa / bb) + 0.5 * bb * torch.log(bb / aa), dim=2)
                except:
                    aa = a_output.view(a_output.shape[0], 1, a_output.shape[1]).repeat(1, b_output.shape[0], 1)
                    dis_matrix[:a_output.shape[0], i * self.args.selection_batch:b.shape[0]] = torch.sum(0.5 * aa * torch.log(aa / bb) + 0.5 * bb * torch.log(bb / aa), dim=2)

        return dis_matrix

    def select(self, **kwargs):
        self.metric = self._metric
        self.run()
        return kCenterGreedy.select(self)
'''