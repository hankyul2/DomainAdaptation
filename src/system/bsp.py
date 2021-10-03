import torch

from src.system.cdan import CDAN_E
from src.system.dann import DANN


def BSP(feature_s, feature_t):
    _, s_s, _ = torch.svd(feature_s)
    _, s_t, _ = torch.svd(feature_t)
    sigma = torch.pow(s_s[0], 2) + torch.pow(s_t[0], 2)
    return sigma


class BSP_DANN(DANN):
    def compute_dc_loss(self, embed_s, embed_t, y_hat_s, y_hat_t):
        bsp_loss = BSP(embed_s, embed_t)
        dc_loss = super(BSP_DANN, self).compute_dc_loss(embed_s, embed_t, y_hat_s, y_hat_t)
        return bsp_loss * 1e-4 + dc_loss


class BSP_CDAN_E(CDAN_E):
    def compute_dc_loss(self, embed_s, embed_t, y_hat_s, y_hat_t):
        bsp_loss = BSP(embed_s, embed_t)
        dc_loss = super(BSP_CDAN_E, self).compute_dc_loss(embed_s, embed_t, y_hat_s, y_hat_t)
        return bsp_loss * 1e-4 + dc_loss
