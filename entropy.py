import torch


def entropy(prediction_softmax, eps=1e-5):
    return -(prediction_softmax * torch.log(prediction_softmax)).sum(dim=1)