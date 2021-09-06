import torch


def entropy(prediction_softmax, eps=1e-5):
    return -(prediction_softmax * torch.log(prediction_softmax+eps)).sum(dim=1)

def divergence(prediction_softmax, eps=1e-5):
    p = prediction_softmax.mean(dim=0)
    return (p * torch.log(p)).sum()