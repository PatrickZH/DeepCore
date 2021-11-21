import numpy as np
from .CoresetMethod import CoresetMethod


class full(CoresetMethod):
    def __init__(self, dst_train, **kwargs):
        self.n_train = len(dst_train)

    def select(self, **kwargs):
        return {"indices": np.arange(self.n_train)}
