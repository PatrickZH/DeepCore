from EarlyTrain import EarlyTrain
import torch
from .. import nets
from forgetting import forgetting
import numpy as np


class SelectionViaProxy(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, selection_method="LeastConfidence",
                 proxy_model="Resnet18", **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, proxy_model)

        selection_choices = ["LeastConfidence",
                             "Entropy",
                             "forgetting"]
        if selection_method not in selection_choices:
            raise NotImplementedError("Selection algorithm unavailable.")
        self.selection_method = selection_method
        self.proxy_model = proxy_model
        if proxy_model not in nets.model_choices:
            raise ValueError("Model unavailable.")

        self.epochs = epochs

    def while_update(self, loss, predicted, targets, epoch, batch_idx, batch_size):
        print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
            epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def finish_train(self):
        with torch.no_grad():
            train_loader = torch.utils.data.DataLoader(self.dst_train, batch_size=self.args.batch)

            scores = np.array([])
            for input, targets in train_loader:

                if self.selection_method == "LeastConfidence":

                    preds = self.model(input.to(self.args.device)).max(axis=1)
                    scores = np.append(scores, -preds)

                elif self.selection_method == "Entropy":
                    preds = torch.nn.functional.softmax(self.model(input.to(self.args.device)), dim=1)
                    scores = np.append(scores, -1 * (np.log(preds) * preds).sum(axis=1))

        return scores

    def select(self):
        if self.selection_method == "forgetting":
            forgetting_selection = forgetting(self.dst_train, self.args, self.fraction, self.random_seed, self.epochs,
                                              self.proxy_model)
            return forgetting_selection.select()
        elif self.selection_method == 'LeastConfidence' or self.selection_method == "Entropy":
            selection_result = np.argsort(self.run())[::-1][:self.coreset_size]
            return torch.utils.data.Subset(self.dst_train, selection_result), selection_result
