from .EarlyTrain import EarlyTrain
import numpy as np
import torch
from .methods_utils import cossim_np, submodular_function, submodular_optimizer


class Similar(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None, balance=False, function="LogDeterminant", greedy="ApproximateLazyGreedy", metric="cossim", **kwargs):
        super(Similar, self).__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)

        if greedy not in submodular_optimizer.optimizer_choices:
            raise ModuleNotFoundError("Greedy optimizer not found.")
        self._greedy = greedy
        self._metric = metric
        self._function = function

        self.balance = balance

    def before_train(self):
        pass

    def after_loss(self, outputs, loss, targets, batch_inds, epoch):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass

    def before_run(self):
        pass

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def calc_gradient(self, index=None):
        '''
        Calculate gradients matrix on current network for specified training dataset.
        '''

        batch_loader = torch.utils.data.DataLoader(
                self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
                batch_size=self.args.selection_batch)
        sample_num = self.n_train if index is None else len(index)

        self.embedding_dim = self.model.get_last_layer().in_features

        # Initialize a matrix to save gradients.
        # (on cpu)
        gradients = np.zeros([sample_num, self.args.num_classes * (self.embedding_dim + 1)], dtype=np.float32)

        for i, (input, targets) in enumerate(batch_loader):
            self.model_optimizer.zero_grad()
            outputs = self.model(input.to(self.args.device))
            loss = self.criterion(torch.nn.functional.softmax(outputs.requires_grad_(True), dim=1),
                                  targets.to(self.args.device)).sum()
            batch_num = targets.shape[0]
            with torch.no_grad():
                bias_parameters_grads = torch.autograd.grad(loss, outputs)[0]
                weight_parameters_grads = self.model.embedding_recorder.embedding.view(batch_num, 1, self.embedding_dim).repeat(1, self.args.num_classes, 1) * bias_parameters_grads.view(batch_num, self.args.num_classes, 1).repeat(1, 1, self.embedding_dim)
                gradients[i * self.args.selection_batch:min((i + 1) * self.args.selection_batch, sample_num)] = torch.cat([bias_parameters_grads, weight_parameters_grads.flatten(1)], dim=1).cpu()
        return gradients

    def finish_run(self):
        # Turn on the embedding recorder and the no_grad flag
        with self.model.embedding_recorder:
            self.model.no_grad = True
            self.train_indx = np.arange(self.n_train)

            if self.balance:
                selection_result = np.array([], dtype=np.int64)
                for c in range(self.num_classes):
                    c_indx = self.train_indx[self.dst_train.targets == c]
                    # Calculate gradients into a matrix
                    gradients = self.calc_gradient(index=c_indx)
                    # Instantiate a submodular function
                    submod_function = submodular_function.__dict__[self._function](index=c_indx, similarity_kernel=lambda a, b:cossim_np(gradients[a], gradients[b]))
                    submod_optimizer = submodular_optimizer.__dict__[self._greedy](args=self.args, index=c_indx, budget=round(self.fraction * len(c_indx)), already_selected=[])

                    c_selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain, update_state=submod_function.update_state)
                    selection_result = np.append(selection_result, c_selection_result)
            else:
                # Calculate gradients into a matrix
                gradients = self.calc_gradient()
                # Instantiate a submodular function
                submod_function = submodular_function.__dict__[self._function](index=self.train_indx,
                                                                               similarity_kernel=lambda a, b: cossim_np(gradients[a], gradients[b]))
                submod_optimizer = submodular_optimizer.__dict__[self._greedy](args=self.args, index=self.train_indx,
                                                                                  budget=self.coreset_size)
                selection_result = submod_optimizer.select(gain_function=submod_function.calc_gain, update_state=submod_function.update_state)

            self.model.no_grad = False
        return selection_result

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result


