from .EarlyTrain import EarlyTrain
from methods_utils.euclidean import euclidean_dist_np
from methods_utils.cossim import cossim_np
import numpy as np
import torch


class Contrastive(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None,
                 balance=True, metric="euclidean", neighbors: int = 10, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)

        self.balance = balance

        assert neighbors > 0 and neighbors < 100
        self.neighbors = neighbors

        if metric == "euclidean":
            self.metric = euclidean_dist_np
        elif metric == "cossim":
            self.metric = lambda a, b: -1. * cossim_np(a, b)
        elif callable(metric):
            self.metric = metric
        else:
            self.metric = euclidean_dist_np

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def find_knn(self, index=None):
        self.model.no_grad = True
        sample_num = self.n_train if index is None else len(index)
        pairwise_matrix = np.zeros([sample_num, sample_num])

        batch_loader = torch.utils.data.DataLoader(
            self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
            batch_size=self.args.selection_batch)

        with self.model.embedding_recorder:
            for i, (aa, _) in enumerate(batch_loader):
                self.model(aa.to(self.args.device))
                aa = self.model.embedding_recorder.embedding.cpu()
                for j, (bb, _) in enumerate(batch_loader):
                    self.model(bb.to(self.args.device))
                    pairwise_matrix[i * self.args.selection_batch:(i+1)*self.args.selection_batch, j * self.args.selection_batch:(j+1)*self.args.selection_batch]=self.metric(aa, self.model.embedding_recorder.embedding.cpu())
        self.model.no_grad = False
        return np.argsort(pairwise_matrix, axis=1)[:, 1:(self.neighbors+1)]

    def calc_kl(self, index=None):
        self.model.no_grad = True
        knn = self.find_knn(index)
        sample_num = self.n_train if index is None else len(index)
        probs = np.zeros([sample_num, self.args.num_classes])

        batch_loader = torch.utils.data.DataLoader(
            self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
            batch_size=self.args.selection_batch)

        for i, (inputs, _) in enumerate(batch_loader):
            probs[i * self.args.selection_batch:(i+1)*self.args.selection_batch] = self.model(inputs.to(self.args.device)).cpu()

        s = np.zeros(sample_num)
        for i in range(0, sample_num, self.args.selection_batch):
            aa = np.expand_dims(probs[i * self.args.selection_batch:(i+1)*self.args.selection_batch], 1).repeat(self.neighbors,1)
            bb = probs[knn[i * self.args.selection_batch:(i+1)*self.args.selection_batch], :]
            s[i * self.args.selection_batch:(i+1)*self.args.selection_batch] = np.mean(np.sum(0.5 * aa * np.log(aa / bb) + 0.5 * bb * np.log(bb / aa), axis=2), axis=1)
        self.model.no_grad = False
        return s

    def finish_run(self):
        if self.balance:
            selection_result = np.array([], dtype=np.int32)
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                selection_result = np.append(selection_result, class_index[self.calc_kl(class_index)[::1][:round(self.fraction*len(class_index))]])
        else:
            selection_result = self.calc_kl()[::1][:self.coreset_size]
        return selection_result


