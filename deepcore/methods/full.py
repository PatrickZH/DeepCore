import numpy as np
from .coresetmethod import CoresetMethod


class Full(CoresetMethod):
    def __init__(self, dst_train, args, fraction, random_seed, **kwargs):
        self.n_train = len(dst_train)

    def select(self, **kwargs):
        return {"indices": np.arange(self.n_train)}
