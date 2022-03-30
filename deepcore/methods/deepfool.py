from .earlytrain import EarlyTrain
import torch
import numpy as np


class DeepFool(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200,
                 specific_model=None, balance: bool = False, max_iter: int = 50, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)

        self.balance = balance
        self.max_iter = max_iter

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    def finish_run(self):
        self.model.no_grad = False

        # Create a data loader for self.dst_train with batch size self.args.selection_batch
        batch_loader = torch.utils.data.DataLoader(self.dst_train, batch_size=self.args.selection_batch
                                                   , num_workers=self.args.workers)

        r = np.zeros(self.n_train, dtype=np.float32)
        batch_num = len(batch_loader)
        for i, (inputs, targets) in enumerate(batch_loader):
            if i % self.args.print_freq == 0:
                print('| Selecting Batch [%3d/%3d]' % (i + 1, batch_num))
            r[(i * self.args.selection_batch):(i * self.args.selection_batch + targets.shape[0])] = self.deep_fool(
                inputs)

        if self.balance:
            selection_result = np.array([], dtype=np.int64)
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                selection_result = np.append(selection_result, class_index[
                    r[class_index].argsort()[:round(len(class_index) * self.fraction)]])
        else:
            selection_result = r.argsort()[:self.coreset_size]
        return {"indices": selection_result, "scores": r}

    def deep_fool(self, inputs):
        # Here, start running DeepFool algorithm.
        self.model.eval()

        # Initialize a boolean mask indicating if selection has been stopped at corresponding positions.
        sample_size = inputs.shape[0]
        boolean_mask = np.ones(sample_size, dtype=bool)
        all_idx = np.arange(sample_size)

        # A matrix to store total pertubations.
        r_tot = np.zeros([sample_size, inputs.shape[1] * inputs.shape[2] * inputs.shape[3]])

        # Set requires_grad for inputs.
        cur_inputs = inputs.requires_grad_(True).to(self.args.device)

        original_shape = inputs.shape[1:]

        # set requires_grad for all parametres in network as False to accelerate autograd
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.model.no_grad = True
        first_preds = self.model(cur_inputs).argmax(dim=1)
        self.model.no_grad = False

        for i in range(self.max_iter):
            f_all = self.model(cur_inputs)

            w_k = []
            for c in range(self.args.num_classes):
                w_k.append(torch.autograd.grad(f_all[:, c].sum(), cur_inputs,
                                               retain_graph=False if c + 1 == self.args.num_classes else True)[
                               0].flatten(1))
            w_k = torch.stack(w_k, dim=0)
            w_k = w_k - w_k[first_preds, boolean_mask[boolean_mask]].unsqueeze(0)
            w_k_norm = w_k.norm(dim=2)

            w_k_norm[first_preds, boolean_mask[
                boolean_mask]] = 1.  # Set w_k_norm for preds positions to 1. to avoid division by zero.

            l_all = (f_all - f_all[boolean_mask[boolean_mask], first_preds].unsqueeze(1)).detach().abs() / w_k_norm.T
            l_all[boolean_mask[
                      boolean_mask], first_preds] = np.inf  # Set l_k for preds positions to inf, as the argmin for each
                                                            # row will be calculated soon.

            l_hat = l_all.argmin(dim=1)
            r_i = l_all[boolean_mask[boolean_mask], l_hat].unsqueeze(1) / w_k_norm[
                l_hat, boolean_mask[boolean_mask]].T.unsqueeze(1) * w_k[l_hat, boolean_mask[boolean_mask]]

            # Update r_tot values.
            r_tot[boolean_mask] += r_i.cpu().numpy()

            cur_inputs += r_i.reshape([r_i.shape[0]] + list(original_shape))

            # Re-input the updated sample into the network and get new predictions.
            self.model.no_grad = True
            preds = self.model(cur_inputs).argmax(dim=1)
            self.model.no_grad = False

            # In DeepFool algorithm, the iteration stops when the updated sample produces a different prediction
            # in the model.
            index_unfinished = (preds == first_preds)
            if torch.all(~index_unfinished):
                break

            cur_inputs = cur_inputs[index_unfinished]
            first_preds = first_preds[index_unfinished]
            boolean_mask[all_idx[boolean_mask][~index_unfinished.cpu().numpy()]] = False

        return (r_tot * r_tot).sum(axis=1)

    def select(self, **kwargs):
        selection_result = self.run()
        return selection_result
