from CoresetMethod import CoresetMethod
import torch


class SelectionViaProxy(CoresetMethod):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, **kwargs):
        super().__init__(dst_train, args, fraction, random_seed)
