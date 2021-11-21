import numpy as np
from .CoresetMethod import CoresetMethod


class uniform(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, balance=False, replace=False, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
        self.balance = balance
        self.replace = replace
        self.n_train = len(dst_train)

    def select_balance(self):
        """The same sampling proportions were used in each class separately."""

        np.random.seed(self.random_seed)
        self.index = np.array([], dtype=np.int64)
        for c in range(self.num_classes):
            self.index = np.append(self.index,
                                   np.random.choice(np.arange(self.n_train)[self.dst_train.targets == c],
                                                    round(self.fraction * sum(self.dst_train.targets == c).item()),
                                                    replace=self.replace))
        return self.index

    def select_no_balance(self):
        np.random.seed(self.random_seed)
        self.index = np.random.choice(np.arange(self.n_train), round(self.n_train * self.fraction),
                                      replace=self.replace)

        return  self.index

    def select(self, **kwargs):
        return {"indices": self.select_balance() if self.balance else self.select_no_balance()}
