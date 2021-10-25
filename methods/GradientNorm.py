from CoresetMethod import CoresetMethod
import torch
from torch import nn
import numpy as np
from .. import nets


class GradientNorm(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)

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

            for index, loss_val in zip(batch_inds, loss):
                loss_val.backward(retain_graph=True)

                # Save gradient of all parameters of the model into one tensor
                self.norm_matrix[index, epoch] = torch.norm(
                    torch.cat([torch.flatten(p.grad) for p in model.parameters() if p.requires_grad]), p=2)

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

        model = nets.__dict__[self.args.model](self.args.channel, self.num_classes).to(self.args.device)
        criterion = nn.CrossEntropyLoss().to(self.args.device)
        criterion.__init__(reduce=False)

        model_optimizer = torch.optim.__dict__[self.args.optimizer](model.parameters(), lr=self.args.lr,
                                                                    momentum=self.args.momentum,
                                                                    weight_decay=self.args.weight_decay)

        # Initialize a matrix to save norms of each sample
        self.norm_matrix = torch.zeros([self.n_train, self.epochs], requires_grad=False).to(self.args.device)

        for epoch in range(self.epochs):
            self.train(model, model_optimizer, criterion, epoch)

    def select(self, **kwargs):
        self.run()
        self.norm_mean = torch.mean(self.norm_mean, dim=1)
        top_k_examples = self.train_indx[np.argsort(self.norm_mean)][:self.coreset_size]
        return torch.utils.data.Subset(self.dst_train, top_k_examples), top_k_examples
