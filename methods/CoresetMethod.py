
class CoresetMethod(object):
    def __init__(self, dst_train, args, fraction=0.5, random_seed=None, **kwargs):
        if fraction <= 0.0 or fraction > 1.0:
            raise ValueError("Illegal Coreset Size.")
        self.dst_train = dst_train
        self.num_classes = len(dst_train.classes)
        self.fraction = fraction
        self.random_seed = random_seed
        self.index = []
        self.args = args

    def select(self, **kwargs):
        return

    def save(self, **kwargs):
        pass


