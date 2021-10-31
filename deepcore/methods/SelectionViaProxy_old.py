from .CoresetMethod import CoresetMethod
import torch
from .. import nets
from forgetting import forgetting
from torch import nn
import numpy as np


class SelectionViaProxy(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, selection_method="LeastConfidence",
                 proxy_model="Resnet18", **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)

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

    def train(self, model, model_optimizer, criterion, epoch):
        """ Train model for one epoch """

        train_loss = 0.
        correct = 0.
        total = 0.

        model.train()

        # Get permutation to shuffle trainset
        trainset_permutation_inds = np.random.permutation(np.arange(self.n_train))

        print('\n=> Training Epoch #%d' % epoch)
        batch_size = self.args.batch

        for batch_idx, batch_start_ind in enumerate(range(0, self.n_train, batch_size)):

            # Get trainset indices for batch
            batch_inds = trainset_permutation_inds[batch_start_ind:
                                                   batch_start_ind + batch_size]

            # Get batch inputs and targets, transform them appropriately
            transformed_trainset = []
            for ind in batch_inds:
                transformed_trainset.append(self.dst_train.__getitem__(ind)[0])
            inputs = torch.stack(transformed_trainset).to(self.args.device)
            targets = torch.LongTensor(np.array(self.dst_train.train_labels)[batch_inds].tolist()).to(self.args.device)

            model_optimizer.zero_grad()

            # Forward propagation, compute loss, get predictions

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Total loss
            loss = loss.mean()
            loss.backward()
            model_optimizer.step()

            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item()))

    def run(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.train_indx = np.arange(self.n_train)

        model = nets.__dict__[self.proxy_model](
            self.args.channel, self.num_classes).to(self.args.device)
        criterion = nn.CrossEntropyLoss().to(self.args.device)
        criterion.__init__(reduce=False)

        model_optimizer = torch.optim.__dict__[self.args.optimizer](model.parameters(), lr=self.args.lr,
                                                                    momentum=self.args.momentum,
                                                                    weight_decay=self.args.weight_decay)
        for epoch in range(self.epochs):
            self.train(model, model_optimizer, criterion, epoch)

        train_loader = torch.utils.data.DataLoader(self.dst_train, batch_size=self.args.batch)

        scores = np.array([])
        for input, targets in train_loader:

            if self.selection_method == "LeastConfidence":

                preds = model(input.to(self.args.device)).max(axis=1)
                scores = np.append(scores, -preds)

            elif self.selection_method == "Entropy":
                preds = torch.nn.functional.softmax(model(input.to(self.args.device)), dim=1)
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
