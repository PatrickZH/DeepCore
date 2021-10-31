from .CoresetMethod import CoresetMethod
import torch
import numpy as np
from torch.utils.data import Subset
from sklearn.metrics import pairwise_distances


# Acknowledgement to
# https://github.com/google/active-learning

# 另一种方法：
# https://github.com/stanford-futuredata/selection-via-proxy/blob/master/svp/common/selection/k_center_greedy.py
# 后续测试选用较好的一种

class kCenterGreedy(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, already_selected=[], metric="euclidean",
                 mapping=lambda x: torch.flatten(x, start_dim=1)):
        super().__init__(dst_train, args, fraction, random_seed)
        self.already_selected = already_selected
        self.metric = metric
        self.mapping = mapping
        self.n_train = len(dst_train)
        self.coreset_size = round(self.n_train * fraction)
        self.min_distances = None

    def update_distances(self, cluster_centers, only_new=True, reset_dist=False):
        if reset_dist:
            self.min_distances = None
        if only_new:
            cluster_centers = [d for d in cluster_centers
                               if d not in self.already_selected]
        if cluster_centers:
            # Update min_distances for all examples given new cluster center.
            x = self.features[cluster_centers]
            dist = pairwise_distances(self.features, x, metric=self.metric)

            if self.min_distances is None:
                self.min_distances = np.min(dist, axis=1).reshape(-1, 1)
            else:
                self.min_distances = np.minimum(self.min_distances, dist)

    def select(self, already_selected=None, **kwargs):
        np.random.seed(self.random_seed)
        try:
            # Assumes that the transform function takes in original data and not
            # flattened data.
            self.features = self.mapping(self.dst_train.train_data)
            self.update_distances(already_selected, only_new=False, reset_dist=True)
        except:
            self.features = torch.flatten(self.dst_train.train_data, start_dim=1)
            self.update_distances(already_selected, only_new=True, reset_dist=False)

        new_batch = []

        for _ in range(self.coreset_size):
            if self.already_selected is None:
                # Initialize centers with a randomly selected datapoint
                ind = np.random.choice(np.arange(self.n_train))
            else:
                ind = np.argmax(self.min_distances)
            # New examples should not be in already selected since those points
            # should have min_distance of zero to a cluster center.
            assert ind not in already_selected

            self.update_distances([ind], only_new=True, reset_dist=False)
            new_batch.append(ind)
        print('Maximum distance from cluster centers is %0.2f'
              % max(self.min_distances))

        self.already_selected = already_selected

        return Subset(self.dst_train, new_batch), new_batch
