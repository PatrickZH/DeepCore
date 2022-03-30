from torch.nn import DataParallel


class MyDataParallel(DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)
    def __setattr__(self, name, value):
        try:
            if name == "no_grad":
                return setattr(self.module, name, value)
            return super().__setattr__(name, value)
        except AttributeError:
            return setattr(self.module, name, value)
