from EarlyTrain import EarlyTrain
import torch
from .. import nets
from forgetting import forgetting
import numpy as np


class uncertainty(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, selection_method="LeastConfidence",
                 specific_model=None, balance=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model)

        selection_choices = ["LeastConfidence",
                             "Entropy",
                             "Margin"]
        if selection_method not in selection_choices:
            raise NotImplementedError("Selection algorithm unavailable.")
        self.selection_method = selection_method

        self.epochs = epochs
        self.balance = balance

    def while_update(self, loss, predicted, targets, epoch, batch_idx, batch_size):
        print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def finish_train(self):
        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                selection_result = np.append(selection_result, np.argsort(self.rank_uncertainty(class_index))[::-1][
                                                               :round(len(class_index) * self.fraction)])
        else:
            selection_result = np.argsort(self.rank_uncertainty())[::-1][:self.coreset_size]
        return selection_result

    def rank_uncertainty(self, index=None):
        with torch.no_grad():
            train_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args.batch)

            scores = np.array([])
            for input, targets in train_loader:

                if self.selection_method == "LeastConfidence":

                    preds = self.model(input.to(self.args.device)).max(axis=1)
                    scores = np.append(scores, -preds)

                elif self.selection_method == "Entropy":
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1)
                    scores = np.append(scores, -1 * (np.log(preds) * preds).sum(axis=1))
                elif self.selection_method == 'Margin':
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1)
                    preds_argmax = torch.argmax(preds, dim=1)
                    max_preds = preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax].clone()
                    preds[torch.ones(preds.shape[0], dtype=bool), preds_argmax] = -1.0
                    preds_sub_argmax = torch.argmax(preds, dim=1)
                    scores = max_preds - preds[torch.ones(preds.shape[0], dtype=bool), preds_sub_argmax]

        return scores

    def select(self, **kwargs):
        selection_result = self.run()
        return torch.utils.data.Subset(self.dst_train, selection_result), selection_result