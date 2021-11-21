from .EarlyTrain import EarlyTrain
from .methods_utils import submodular_optimizer
import torch
import numpy as np


class Glister(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None,
                 balance: bool = True, greedy="LazyGreedy", eta=None, dst_val=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)

        self.balance = balance
        self.eta = args.lr if eta is None else eta

        self.dst_val = dst_train if dst_val is None else dst_val
        self.n_val = len(self.dst_val)

        if greedy not in submodular_optimizer.optimizer_choices:
            raise ModuleNotFoundError("Greedy optimizer not found.")
        self._greedy = greedy

    def calc_gradient(self, index=None, val=False, record_val_detail=False):
        '''
        Calculate gradients matrix on current network for training or validation dataset.
        :param index: indices of data for gradients
        :param val:
        :param record_val_detail:
        :return:
        '''

        if val:
            batch_loader = torch.utils.data.DataLoader(
                self.dst_val if index is None else torch.utils.data.Subset(self.dst_val, index),
                batch_size=self.args.selection_batch)
            sample_num = len(self.dst_val.targets) if index is None else len(index)
        else:
            batch_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args.selection_batch)
            sample_num = self.n_train if index is None else len(index)

        self.embedding_dim = self.model.get_last_layer().in_features
        gradients = []
        if val and record_val_detail:
            self.init_out = []
            self.init_emb = []
            self.init_y = []
            # self.init_out = torch.zeros([sample_num, self.args.num_classes], requires_grad=False).to(self.args.device)
            # self.init_emb = torch.zeros([sample_num, self.embedding_dim], requires_grad=False).to(self.args.device)
            # self.init_y = torch.zeros([sample_num], requires_grad=False, dtype=torch.long).to(self.args.device)
        for i, (input, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device))
            loss = self.criterion(torch.nn.functional.softmax(outputs.requires_grad_(True), dim=1), targets.to(self.args.device)).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                weight_parameters_grads = self.model.embedding_recorder.embedding.view(batch_num, 1, self.embedding_dim).repeat(1,
                                                                                                               self.args.num_classes,
                                                                                                               1) * bias_parameters_grads.view(
                    batch_num, self.args.num_classes, 1).repeat(1, 1, self.embedding_dim)
                gradients.append(torch.cat(
                    [bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1).cpu())

                if val and record_val_detail:
                    self.init_out.append(outputs.cpu())
                    self.init_emb.append(self.model.embedding_recorder.embedding.cpu())
                    self.init_y.append(targets)
                    # self.init_out[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = outputs
                    # self.init_emb[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = self.model.embedding
                    # self.init_y[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = targets.to(
                    #     self.args.device)
        gradients = torch.cat(gradients, dim=0)
        if val:
            self.val_grads = torch.mean(gradients, dim=0)
            if self.dst_val == self.dst_train:
                # No validation set was provided while instantiating Glister, so self.dst_val == self.dst_train
                self.train_grads = gradients
        else:
            self.train_grads = gradients
        if val and record_val_detail:
            with torch.no_grad():
                self.init_out = torch.cat(self.init_out, dim=0)
                self.init_emb = torch.cat(self.init_emb, dim=0)
                self.init_y = torch.cat(self.init_y)

    def update_val_gradients(self, new_selection, selected_for_train):

        sum_selected_train_gradients = torch.mean(self.train_grads[selected_for_train], dim=0)

        new_outputs = self.init_out - self.eta * sum_selected_train_gradients[:self.args.num_classes].view(1,-1).repeat(self.init_out.shape[0], 1) - self.eta * torch.matmul(self.init_emb, sum_selected_train_gradients[self.args.num_classes:].view(self.args.num_classes, -1).T)

        sample_num = new_outputs.shape[0]
        gradients = torch.zeros([sample_num, self.args.num_classes * (self.embedding_dim + 1)], requires_grad=False)
        i = 0
        while i * self.args.selection_batch < sample_num:
            batch_indx = np.arange(sample_num)[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)]
            new_out_puts_batch = new_outputs[batch_indx].clone().detach().requires_grad_(True)
            loss = self.criterion(torch.nn.functional.softmax(new_out_puts_batch, dim=1), self.init_y[batch_indx])
            batch_num = len(batch_indx)
            bias_parameters_grads = torch.autograd.grad(loss.sum(), new_out_puts_batch, retain_graph=True)[0]

            weight_parameters_grads = self.init_emb[batch_indx].view(batch_num, 1, self.embedding_dim).repeat(1, self.args.num_classes, 1) * bias_parameters_grads.view(batch_num, self.args.num_classes, 1).repeat(1, 1, self.embedding_dim)
            gradients[batch_indx] = torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1).cpu()
            i += 1

        self.val_grads = torch.mean(gradients, dim=0)

    def finish_run(self):
        self.model.embedding_recorder.record_embedding = True
        self.model.no_grad = True

        self.train_indx = np.arange(self.n_train)
        self.val_indx = np.arange(self.n_val)
        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            #weights = np.array([], dtype=np.float32)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                c_val_inx = self.val_indx[self.dst_val.targets == c]
                self.calc_gradient(index=c_val_inx, val=True, record_val_detail=True)
                if self.dst_val != self.dst_train:
                    self.calc_gradient(index=c_indx)
                submod_optimizer = submodular_optimizer.__dict__[self._greedy](args=self.args, index=c_indx, budget=round(self.fraction * len(c_indx)))
                c_selection_result = submod_optimizer.select(gain_function=lambda idx_gain, selected, **kwargs: torch.matmul(self.train_grads[idx_gain], self.val_grads.view(-1, 1)).detach().cpu().numpy().flatten(), upadate_state=self.update_val_gradients)
                selection_result = np.append(selection_result, c_selection_result)

        else:
            self.calc_gradient(val=True, record_val_detail=True)
            if self.dst_val != self.dst_train:
                self.calc_gradient()
            #selection_result = self.greedy_select(index=np.arange(self.n_train), budget=self.coreset_size)
            submod_optimizer = submodular_optimizer.__dict__[self._greedy](args=self.args, index=np.arange(self.n_train), budget=self.coreset_size)
            selection_result = submod_optimizer.select(gain_function=lambda idx_gain, selected, **kwargs: torch.matmul(self.train_grads[idx_gain], self.val_grads.view(-1, 1)).detach().cpu().numpy().flatten(), upadate_state=self.update_val_gradients)

        self.model.embedding_recorder.record_embedding = False
        self.model.no_grad = False
        return {"indices": selection_result}

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))


"""from .CoresetMethod import CoresetMethod
import torch
import numpy as np


class Glister(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, network=None, optimizer=None, criterion=None,
                 balance=True, eta=None, dst_val=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.balance = balance
        self.eta = args.lr if eta is None else eta

        if network is None or optimizer is None or criterion is None:
            raise ValueError("Network, criterion or optimizer is not specified.")
        self.network = network
        self.optimizer = optimizer
        self.criterion = criterion

        self.dst_val = dst_train if dst_val is None else dst_val
        self.n_val = len(self.dst_val)

    def calc_gradient(self, index=None, val=False, record_val_detail=False):


        if val:
            batch_loader = torch.utils.data.DataLoader(
                self.dst_val if index is None else torch.utils.data.Subset(self.dst_val, index),
                batch_size=self.args.selection_batch)
            sample_num = len(self.dst_val.targets) if index is None else len(index)
        else:
            batch_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args.selection_batch)
            sample_num = self.n_train if index is None else len(index)

        self.embedding_dim = self.network.get_last_layer().in_features
        gradients = torch.zeros([sample_num, self.args.num_classes * (self.embedding_dim + 1)], requires_grad=False, device='cpu')#.to(self.args.device)
        if val and record_val_detail:
            self.init_out = []
            self.init_emb = []
            self.init_y = []
            # self.init_out = torch.zeros([sample_num, self.args.num_classes], requires_grad=False).to(self.args.device)
            # self.init_emb = torch.zeros([sample_num, self.embedding_dim], requires_grad=False).to(self.args.device)
            # self.init_y = torch.zeros([sample_num], requires_grad=False, dtype=torch.long).to(self.args.device)
        for i, (input, targets) in enumerate(batch_loader):
            self.optimizer.zero_grad()
            outputs = self.network(input.to(self.args.device))
            loss = self.criterion(torch.nn.functional.softmax(outputs, dim=1), targets.to(self.args.device))
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss.sum(), outputs)[0].cpu()
                weight_parameters_grads = self.network.embedding.cpu().view(batch_num, 1, self.embedding_dim).repeat(1,
                                                                                                               self.args.num_classes,
                                                                                                               1) * bias_parameters_grads.view(
                    batch_num, self.args.num_classes, 1).repeat(1, 1, self.embedding_dim)
                gradients[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = torch.cat(
                    [bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1)

                if val and record_val_detail:
                    self.init_out.append(outputs.cpu())
                    self.init_emb.append(self.network.embedding.cpu())
                    self.init_y.append(targets)#.to(self.args.device))
                    # self.init_out[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = outputs
                    # self.init_emb[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = self.network.embedding
                    # self.init_y[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = targets.to(
                    #     self.args.device)
                #del self.network.embedding
        if val:
            self.val_grads = torch.mean(gradients, dim=0)
        else:
            self.train_grads = gradients
        if val and record_val_detail:
            self.init_out = torch.cat(self.init_out, dim=0)
            self.init_emb = torch.cat(self.init_emb, dim=0)
            self.init_y = torch.cat(self.init_y)

    def greedy_select(self, index, budget: int):
        selected = np.zeros(len(index), dtype=bool)

        greedy_gain = np.zeros(len(index))
        while sum(selected) < budget:
            if sum(selected) % self.args.print_freq == 0:
                print("| Selecting [%3d/%3d]" % (sum(selected) + 1, budget))
            greedy_gain[~selected] = torch.matmul(self.train_grads[~selected], self.val_grads.view(-1, 1)).detach().cpu().numpy().flatten()
            current_selection = greedy_gain.argmax().item()
            selected[current_selection] = True
            greedy_gain[current_selection] = -1.
            if sum(selected) < budget:
                self.update_val_gradients(selected)
        return index[selected], [1] * budget

    def update_val_gradients(self, selected_for_train):

        sum_selected_train_gradients = torch.mean(self.train_grads[selected_for_train], dim=0)

        new_outputs = self.init_out - self.eta * sum_selected_train_gradients[:self.args.num_classes].view(1,-1).repeat(self.init_out.shape[0], 1) - self.eta * torch.matmul(self.init_emb, sum_selected_train_gradients[self.args.num_classes:].view(self.args.num_classes, -1).T)

        sample_num = new_outputs.shape[0]
        gradients = torch.zeros([sample_num, self.args.num_classes * (self.embedding_dim + 1)], requires_grad=False, device='cpu')#.to(self.args.device)

        i = 0
        while i * self.args.selection_batch < sample_num:
            batch_indx = np.arange(sample_num)[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)]
            new_out_puts_batch = new_outputs[batch_indx].clone().detach().requires_grad_(True)
            loss = self.criterion(torch.nn.functional.softmax(new_out_puts_batch, dim=1), self.init_y[batch_indx])
            batch_num = len(batch_indx)
            bias_parameters_grads = torch.autograd.grad(loss.sum(), new_out_puts_batch)[0]

            weight_parameters_grads = self.init_emb[batch_indx].view(batch_num, 1, self.embedding_dim).repeat(1, self.args.num_classes, 1) * bias_parameters_grads.view(batch_num, self.args.num_classes, 1).repeat(1, 1, self.embedding_dim)
            gradients[batch_indx] = torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1)
            i += 1

        self.val_grads = torch.mean(gradients, dim=0)

    def select(self, **kwargs):
        self.network.record_embedding = True
        self.network.no_grad = True

        self.train_indx = np.arange(self.n_train)
        self.val_indx = np.arange(self.n_val)
        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            weights = np.array([], dtype=np.float32)
            for c in range(self.num_classes):
                c_indx = self.train_indx[self.dst_train.targets == c]
                self.calc_gradient(index=c_indx)
                c_val_inx = self.val_indx[self.dst_val.targets == c]
                self.calc_gradient(index=c_val_inx, val=True, record_val_detail=True)
                c_selection_result, c_weights = self.greedy_select(index=c_indx,
                                                                   budget=round(self.fraction * len(c_indx)))
                selection_result = np.append(selection_result, c_selection_result)
                weights = np.append(weights, c_weights)

        else:
            self.calc_gradient()
            self.calc_gradient(val=True, record_val_detail=True)
            selection_result, weights = self.greedy_select(index=np.arange(self.n_train), budget=self.coreset_size)

        self.network.record_embedding = False
        self.network.no_grad = False
        return selection_result
"""