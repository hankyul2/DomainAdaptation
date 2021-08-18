import torch


def BSP(feature_s, feature_t):
    _, s_s, _ = torch.svd(feature_s)
    _, s_t, _ = torch.svd(feature_t)
    sigma = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
    return sigma

if __name__ == '__main__':
    num_batch = 32
    num_class = 10
    source_feature = torch.rand(num_batch, num_class)
    target_feature = torch.rand(num_batch, num_class)
    bsp_loss = BSP(source_feature, target_feature)