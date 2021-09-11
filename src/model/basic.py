import torch
from torch import nn
import torch.nn.functional as F


class BasicModel(nn.Module):
    def __init__(self, backbone, fc_dim=2048, embed_dim=256, nclass=31, dropout=0.1):
        super().__init__()
        self.backbone = backbone
        self.bottleneck = nn.Linear(fc_dim, embed_dim)
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=dropout)
        self.feature_extractor = lambda x: self.dropout(F.tanh(self.bottleneck(self.backbone(x))))
        self.fc = nn.Linear(embed_dim, nclass)

    def forward(self, x):
        feature = self.feature_extractor(x)
        class_prediction = self.fc(feature)
        return class_prediction

    def predict(self, x):
        feature = self.feature_extractor(x)
        class_prediction = self.fc(feature)
        return class_prediction


def get_basic_model(backbone, nclass=31, fc_dim=2048, embed_dim=256):
    model = BasicModel(backbone, fc_dim=fc_dim, embed_dim=embed_dim, nclass=nclass)

    for name, param in model.named_parameters():
        if 'backbone' in name or 'bn' in name:
            continue
        if param.dim() > 1:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.constant_(param, 0)

    return model