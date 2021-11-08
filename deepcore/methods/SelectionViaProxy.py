from .uncertainty import uncertainty
from .. import nets
from .forgetting import forgetting


def SelectionViaProxy(dst_train, args, fraction=0.5, random_seed=None, epochs=200, selection_method="LeastConfidence", proxy_model="ResNet18", balance=True, **kwargs):
    selection_choices = ["LeastConfidence",
                         "Entropy",
                         "forgetting",
                         'Margin']
    if selection_method not in selection_choices:
        raise NotImplementedError("Selection algorithm unavailable.")

    if proxy_model not in nets.model_choices:
        raise ValueError("Model unavailable.")

    if selection_method == "forgetting":
        return forgetting(dst_train, args, fraction, random_seed, epochs, proxy_model, balance)
    else:
        return uncertainty(dst_train, args, fraction, random_seed, epochs, selection_method, proxy_model, balance, **kwargs)


