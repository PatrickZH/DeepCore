from .kcentergreedy import kCenterGreedy
import torch


# Acknowlegement to:
# https://github.com/sharat29ag/CDAL


class ContextualDiversity(kCenterGreedy):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200,
                 specific_model=None, balance=True, already_selected=[], torchvision_pretrain: bool = False, **kwargs):
        super(ContextualDiversity, self).__init__(dst_train, args, fraction, random_seed, epochs=epochs, specific_model=specific_model, balance=balance, already_selected=already_selected, torchvision_pretrain=torchvision_pretrain, **kwargs)
        self.metric = self._metric

    def _metric(self, a_output, b_output):
        with torch.no_grad():
            # Overload self.metric function for kCenterGreedy Algorithm
            aa = a_output.view(a_output.shape[0], 1, a_output.shape[1]).repeat(1, b_output.shape[0], 1)
            bb = b_output.view(1, b_output.shape[0], b_output.shape[1]).repeat(a_output.shape[0], 1, 1)
            return torch.sum(0.5 * aa * torch.log(aa / bb) + 0.5 * bb * torch.log(bb / aa), dim=2)

    def construct_matrix(self, index=None):
        self.model.eval()
        self.model.no_grad = True
        sample_num = self.n_train if index is None else len(index)
        matrix = torch.zeros([sample_num, self.args.num_classes], requires_grad=False).to(self.args.device)
        batch_loader = torch.utils.data.DataLoader(self.dst_train if index is None else
                            torch.utils.data.Subset(self.dst_train, index), batch_size=self.args.selection_batch
                                                   ,num_workers=self.args.workers)
        for i, (inputs, _) in enumerate(batch_loader):
            matrix[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = torch.nn.functional.softmax(self.model(inputs.to(self.args.device)), dim=1)
        self.model.no_grad = False
        return matrix
