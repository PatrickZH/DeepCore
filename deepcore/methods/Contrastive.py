from .EarlyTrain import EarlyTrain
from .methods_utils.euclidean import euclidean_dist_pair_np
from .methods_utils.cossim import cossim_pair_np
import numpy as np
import torch
from .. import nets
from copy import deepcopy
from torchvision import transforms


class Contrastive(EarlyTrain):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, epochs=200, specific_model=None,
                 balance=True, metric="euclidean", neighbors: int = 10, pretrain_model: str = "ResNet18", **kwargs):
        super().__init__(dst_train, args, fraction, random_seed, epochs, specific_model, **kwargs)

        self.balance = balance

        assert neighbors > 0 and neighbors < 100
        self.neighbors = neighbors

        if metric == "euclidean":
            self.metric = euclidean_dist_pair_np
        elif metric == "cossim":
            self.metric = lambda a, b: -1. * cossim_pair_np(a, b)
        elif callable(metric):
            self.metric = metric
        else:
            self.metric = euclidean_dist_pair_np

        assert pretrain_model in nets.model_choices
        self.pretrain_model = pretrain_model

    def num_classes_mismatch(self):
        raise ValueError("num_classes of pretrain dataset does not match that of the training dataset.")

    def while_update(self, outputs, loss, targets, epoch, batch_idx, batch_size):
        if batch_idx % self.args.print_freq == 0:
            print('| Epoch [%3d/%3d] Iter[%3d/%3d]\t\tLoss: %.4f' % (
                epoch, self.epochs, batch_idx + 1, (self.n_pretrain_size // batch_size) + 1, loss.item()))

    '''
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
                aa = self.model.embedding_recorder.embedding.flatten(1).cpu().numpy()
                for j, (bb, _) in enumerate(batch_loader):
                    self.model(bb.to(self.args.device))
                    pairwise_matrix[i * self.args.selection_batch:(i+1)*self.args.selection_batch, j * self.args.selection_batch:(j+1)*self.args.selection_batch]=self.metric(aa, self.model.embedding_recorder.embedding.flatten(1).cpu().numpy())
        self.model.no_grad = False
        return np.argsort(pairwise_matrix, axis=1)[:, 1:(self.neighbors+1)]


    def find_knn(self, index=None):
        self.model.no_grad = True
        sample_num = self.n_train if index is None else len(index)
        embdeddings = []

        batch_loader = torch.utils.data.DataLoader(
            self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
            batch_size=self.args.selection_batch)

        with self.model.embedding_recorder:
            for aa, _ in batch_loader:
                self.model(aa.to(self.args.device))
                embdeddings.append(self.model.embedding_recorder.embedding.flatten(1).cpu().numpy())

        self.model.no_grad = False

        return np.argsort(self.metric(np.concatenate(embdeddings, axis=0)), axis=1)[:, 1:(self.neighbors+1)]
    '''

    def find_knn(self):
        """
        Find k-nearest-neighbor data points with the pretrained embedding model
        :return: knn matrix
        """

        # Initialize pretrained model
        model = nets.__dict__[self.pretrain_model](channel=self.args.channel, num_classes=self.args.num_classes,
                                                   im_size=(224, 224), record_embedding=True, no_grad=True,
                                                   pretrained=True).to(self.args.device)

        # Resize dst_train to 224*224
        if self.args.im_size[0] != 224 or self.args.im_size[1] != 224:
            dst_train = deepcopy(self.dst_train)
            dst_train.transform = transforms.Compose([dst_train.transform, transforms.Resize(224)])
        else:
            dst_train = self.dst_train

        # Start recording embedding vectors
        embdeddings = []
        batch_loader = torch.utils.data.DataLoader(dst_train, batch_size=self.args.selection_batch)
        batch_num = len(batch_loader)

        for i, (aa, _) in enumerate(batch_loader):
            if i % self.args.print_freq == 0:
                print("| Caculating embeddings for batch [%3d/%3d]" % (i + 1, batch_num))
            model(aa.to(self.args.device))
            embdeddings.append(model.embedding_recorder.embedding.flatten(1).cpu().numpy())
        embdeddings = np.concatenate(embdeddings, axis=0)

        # Calculate the distance matrix and return knn results
        if self.balance:
            knn = []
            for c in range(self.args.num_classes):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                knn.append(np.argsort(self.metric(embdeddings[class_index]), axis=1)[:, 1:(self.neighbors + 1)])
            return knn
        else:
            return np.argsort(self.metric(embdeddings), axis=1)[:, 1:(self.neighbors + 1)]

    def calc_kl(self, knn, index=None):
        self.model.no_grad = True
        sample_num = self.n_train if index is None else len(index)
        probs = np.zeros([sample_num, self.args.num_classes])

        batch_loader = torch.utils.data.DataLoader(
            self.dst_train if index is None else torch.utils.data.Subset(self.dst_train, index),
            batch_size=self.args.selection_batch)
        batch_num = len(batch_loader)

        for i, (inputs, _) in enumerate(batch_loader):
            probs[i * self.args.selection_batch:(i + 1) * self.args.selection_batch] = torch.nn.functional.softmax(
                self.model(inputs.to(self.args.device)), dim=1).cpu()

        s = np.zeros(sample_num)
        for i in range(0, sample_num, self.args.selection_batch):
            if i % self.args.print_freq == 0:
                print("| Caculating KL-divergence for batch [%3d/%3d]" % (i // self.args.selection_batch + 1, batch_num))
            aa = np.expand_dims(probs[i:(i + self.args.selection_batch)], 1).repeat(self.neighbors, 1)
            bb = probs[knn[i:(i + self.args.selection_batch)], :]
            s[i:(i + self.args.selection_batch)] = np.mean(
                np.sum(0.5 * aa * np.log(aa / bb) + 0.5 * bb * np.log(bb / aa), axis=2), axis=1)
        self.model.no_grad = False
        return s

    def finish_run(self):
        if self.balance:
            selection_result = np.array([], dtype=np.int32)
            for c, knn in zip(range(self.args.num_classes), self.knn):
                class_index = np.arange(self.n_train)[self.dst_train.targets == c]
                selection_result = np.append(selection_result, class_index[np.argsort(
                    self.calc_kl(knn, class_index))[::1][:round(self.fraction * len(class_index))]])
        else:
            selection_result = np.argsort(self.calc_kl(self.knn))[::1][:self.coreset_size]
        return {"indices": selection_result}

    def select(self, **kwargs):
        self.knn = self.find_knn()
        selection_result = self.run()
        return selection_result