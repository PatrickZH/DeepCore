from .CoresetMethod import CoresetMethod
import torch, time
from torch import nn
import numpy as np
from copy import deepcopy
from .. import nets
from torchvision import transforms


class EarlyTrain(CoresetMethod):
    '''
    Core code for training related to coreset selection methods when pre-training is required.

    :param
    dst_train:

    '''
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None,
                 torchvision_pretrain: bool = False, dst_pretrain_dict: dict = {}, fraction_pretrain=1., dst_test=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.epochs = epochs
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.specific_model = specific_model

        if specific_model is not None and specific_model not in nets.model_choices:
            self.specific_model = None

        if fraction_pretrain <= 0. or fraction_pretrain > 1.:
            raise ValueError("Illegal pretrain fraction value.")
        self.fraction_pretrain = fraction_pretrain

        if dst_pretrain_dict.__len__() != 0:
            dict_keys = dst_pretrain_dict.keys()
            if 'im_size' not in dict_keys or 'channel' not in dict_keys or 'dst_train' not in dict_keys or 'num_classes' not in dict_keys:
                raise AttributeError(
                    'Argument dst_pretrain_dict must contain imszie, channel, dst_train and num_classes.')
            if dst_pretrain_dict['im_size'][0] != args.im_size[0] or dst_pretrain_dict['im_size'][0] != args.im_size[0]:
                raise ValueError("im_size of pretrain dataset does not match that of the training dataset.")
            if dst_pretrain_dict['channel'] != args.channel:
                raise ValueError("channel of pretrain dataset does not match that of the training dataset.")
            if dst_pretrain_dict['num_classes'] != args.num_classes:
                self.num_classes_mismatch()

        self.dst_pretrain_dict = dst_pretrain_dict
        self.torchvision_pretrain = torchvision_pretrain
        self.if_dst_pretrain = (len(self.dst_pretrain_dict) != 0)

        if torchvision_pretrain:
            # Pretrained models in torchvision only accept 224*224 inputs, therefore we resize current datasets to 224*224.
            if args.im_size[0] != 224 or args.im_size[1] != 224:
                self.dst_train = deepcopy(dst_train)
                self.dst_train.transform = transforms.Compose([self.dst_train.transform, transforms.Resize(224)])
                if self.if_dst_pretrain:
                    self.dst_pretrain_dict['dst_train'] = deepcopy(dst_pretrain_dict['dst_train'])
                    self.dst_pretrain_dict['dst_train'].transform = transforms.Compose(
                        [self.dst_pretrain_dict['dst_train'].transform, transforms.Resize(224)])
        if self.if_dst_pretrain:
            self.n_pretrain = len(self.dst_pretrain_dict['dst_train'])
        self.n_pretrain_size = round(
            self.fraction_pretrain * (self.n_pretrain if self.if_dst_pretrain else self.n_train))
        self.dst_test = dst_test

    def train(self, epoch, list_of_train_idx, **kwargs):
        """ Train model for one epoch """

        self.before_train()

        self.model.train()

        # Get permutation to shuffle trainset
        trainset_permutation_inds = np.random.permutation(list_of_train_idx)

        print('\n=> Training Epoch #%d' % epoch)
        batch_size = self.args.selection_batch

        for batch_idx, batch_start_ind in enumerate(range(0, self.n_pretrain_size, batch_size)):
            # Get trainset indices for batch
            batch_inds = trainset_permutation_inds[batch_start_ind:
                                                   batch_start_ind + batch_size]

            # Get batch inputs and targets, transform them appropriately
            transformed_trainset = []
            transformed_taregts = []
            for ind in batch_inds:
                batch_data_and_targets = (
                    self.dst_pretrain_dict['dst_train'] if self.if_dst_pretrain else self.dst_train).__getitem__(ind)
                transformed_trainset.append(batch_data_and_targets[0])
                transformed_taregts.append(batch_data_and_targets[1])
            inputs = torch.stack(transformed_trainset).to(self.args.device)
            targets = torch.tensor(transformed_taregts, device=self.args.device, requires_grad=False, dtype=torch.long)

            # Forward propagation, compute loss, get predictions
            self.model_optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)

            self.after_loss(outputs, loss, targets, batch_inds, epoch)

            # Update loss, backward propagate, update optimizer
            loss = loss.mean()

            self.while_update(outputs, loss, targets, epoch, batch_idx, batch_size)

            loss.backward()
            self.model_optimizer.step()
        return self.finish_train()

    def run(self):
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        self.train_indx = np.arange(self.n_train)

        # Setup model and loss
        self.model = nets.__dict__[self.args.model if self.specific_model is None else self.specific_model](
            self.args.channel, self.dst_pretrain_dict["num_classes"] if self.if_dst_pretrain else self.num_classes,
            pretrained=self.torchvision_pretrain,
            im_size=(224, 224) if self.torchvision_pretrain else self.args.im_size).to(self.args.device)
        self.criterion = nn.CrossEntropyLoss().to(self.args.device)
        self.criterion.__init__()

        # Setup optimizer
        if self.args.selection_optimizer == "SGD":
            self.model_optimizer = torch.optim.SGD(self.model.parameters(), lr=self.args.selection_lr,
                                                   momentum=self.args.selection_momentum,
                                                   weight_decay=self.args.selection_weight_decay,
                                                   nesterov=self.args.selection_nesterov)
        elif self.args.selection_optimizer == "Adam":
            self.model_optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.selection_lr, weight_decay=self.args.selection_weight_decay)
        else:
            self.model_optimizer = torch.optim.__dict__[self.args.selection_optimizer](self.model.parameters(), lr=self.args.selection_lr,
                                                                         momentum=self.args.selection_momentum,
                                                                         weight_decay=self.args.selection_weight_decay,
                                                                      nesterov=self.args.selection_nesterov)

        self.before_run()

        if self.args.bb!="":
            self.model.load_state_dict(torch.load(self.args.bb, map_location=self.args.device))

        for epoch in range(self.epochs):
            list_of_train_idx = np.random.choice(np.arange(self.n_pretrain if self.if_dst_pretrain else self.n_train),
                                                 self.n_pretrain_size, replace=False)
            self.before_epoch()
            self.train(epoch, list_of_train_idx)
            if self.dst_test is not None and self.args.selection_test_interval > 0 and (epoch+1) % self.args.selection_test_interval == 0:
                self.test(epoch)
            self.after_epoch()


        if self.args.aa!="":
            torch.save(self.model.state_dict(), self.args.aa)

        return self.finish_run()

    def test(self, epoch):
        self.model.no_grad = True
        self.model.eval()

        test_loader = torch.utils.data.DataLoader(self.dst_test if self.args.selection_test_fraction==1. else
                    torch.utils.data.Subset(self.dst_test, np.random.choice(np.arange(len(self.dst_test)),
                  round(len(self.dst_test) * self.args.selection_test_fraction), replace=False)),
                    batch_size=self.args.selection_batch, shuffle=True, num_workers=2, pin_memory=True)
        correct = 0.
        total = 0.

        print('\n=> Testing Epoch #%d' % epoch)

        for batch_idx, (input, target) in enumerate(test_loader):
            output = self.model(input.to(self.args.device))
            loss = self.criterion(output, target.to(self.args.device)).sum()

            predicted = torch.max(output.data, 1).indices.cpu()
            correct += predicted.eq(target).sum().item()
            total += target.size(0)

            if batch_idx % self.args.print_freq == 0:
                print('| Test Epoch [%3d/%3d] Iter[%3d/%3d]\t\tTest Loss: %.4f Test Acc: %.3f%%' % (
            epoch, self.epochs, batch_idx + 1, (round(len(self.dst_test) * self.args.selection_test_fraction) //
                                                self.args.selection_batch) + 1, loss.item(), 100. * correct / total))

        self.model.no_grad = False

    def num_classes_mismatch(self):
        pass

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
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

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result
