import torch

def make_optimizer(cfg, model):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = 0.01
        weight_decay = 0.0001

        if "bias" in key:
            lr = lr * 2
            weight_decay = 0
        if key.endswith(".offset.weight") or key.endswith(".offset.bias"):
            lr *= 1

        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr, momentum=0.9)
    return optimizer
