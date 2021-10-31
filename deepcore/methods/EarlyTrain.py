from .CoresetMethod import CoresetMethod
import torch, time
from torch import nn
import numpy as np
from .. import nets


class EarlyTrain(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model

        if specific_model is not None and specific_model not in nets.model_choices:
            self.specific_model = None

    '''
    def before_train(self):
        self.train_loss = 0.
        self.correct = 0.
        self.total = 0.


    def after_loss(self, outputs, loss, predicted, targets, batch_inds):
        # Update statistics and loss
        acc = predicted == targets
        for j, index in enumerate(batch_inds):

            # Get index in original dataset (not sorted by forgetting)
            index_in_original_dataset = self.train_indx[index]

            # Compute missclassification margin
            output_correct_class = outputs.data[j, targets[j].item()]
            sorted_output, _ = torch.sort(outputs.data[j, :])
            if acc[j]:
                # Example classified correctly, highest incorrect class is 2nd largest output
                output_highest_incorrect_class = sorted_output[-2]
            else:
                # Example misclassified, highest incorrect class is max output
                output_highest_incorrect_class = sorted_output[-1]
            margin = output_correct_class.item(
            ) - output_highest_incorrect_class.item()

            # Add the statistics of the current training example to dictionary
            index_stats = self.example_stats.get(index_in_original_dataset,
                                                 [[], [], []])
            index_stats[0].append(loss[j].item())
            index_stats[1].append(acc[j].sum().item())
            index_stats[2].append(margin)
            self.example_stats[index_in_original_dataset] = index_stats

    def while_update(self, loss, predicted, targets, epoch, batch_idx, batch_size):
        self.train_loss += loss.item()
        self.total += targets.size(0)
        self.correct += predicted.eq(targets.data).cpu().sum()

        print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f Acc@1: %.3f%%' % (
            epoch, self.epochs, batch_idx + 1, (self.n_train // batch_size) + 1, loss.item(),
            100. * self.correct.item() / self.total))

        # Add training accuracy to dict
        index_stats = self.example_stats.get('train', [[], []])
        index_stats[1].append(100. * self.correct.item() / float(self.total))
        self.example_stats['train'] = index_stats

    def finish_train(self):
        pass
    
    def before_epoch(self):
        self.start_time = time.time()

    def after_epoch(self):
        epoch_time = time.time() - self.start_time
        self.elapsed_time += epoch_time
        print('| Elapsed time : %d:%02d:%02d' % (self.get_hms(self.elapsed_time)))

    def before_run(self):
        self.best_acc = 0
        self.elapsed_time = 0
    '''

    def train(self, epoch, **kwargs):
        """ Train model for one epoch """

        self.before_train()

        self.model.train()

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
            targets = torch.LongTensor(np.array(self.dst_train.targets)[batch_inds].tolist()).to(self.args.device)

            # Forward propagation, compute loss, get predictions
            self.model_optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(torch.nn.functional.softmax(outputs, dim=1), targets)
            _, predicted = torch.max(outputs.data, 1)

            self.after_loss(outputs, loss, predicted, targets, batch_inds, epoch)

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()

            self.while_update(loss, predicted, targets, epoch, batch_idx, batch_size)

            loss.backward()
            self.model_optimizer.step()
        return self.finish_train()

    def run(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.train_indx = np.arange(self.n_train)

        # Setup model and loss
        self.model = nets.__dict__[self.args.model if self.specific_model is None else self.specific_model](
            self.args.channel, self.num_classes).to(self.args.device)
        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.criterion.__init__(reduce=False)

        # Setup optimizer
        self.model_optimizer = torch.optim.__dict__[self.args.optimizer](self.model.parameters(), lr=self.args.lr,
                                                                         momentum=self.args.momentum,
                                                                         weight_decay=self.args.weight_decay)

        self.before_run()

        for epoch in range(self.epochs):
            self.before_epoch()
            self.train(epoch)
            # self.test(epoch, model)
            self.after_epoch()
        return self.finish_run()

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, predicted, targets, batch_inds, epoch):
        pass

    def while_update(self, loss, predicted, targets, epoch, batch_idx, batch_size):
        pass

    def finish_train(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def finish_run(self):
        pass
