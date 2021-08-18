from torch import nn
import torch.nn.functional as F
from src.grl import GRL


class DomainClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout=0.5):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        x = self.layer(x)
        return x


class CDAN(nn.Module):
    def __init__(self, backbone, fc_dim=2048, embed_dim=256, nclass=31, hidden_dim=1024):
        super().__init__()
        self.backbone = backbone
        self.bottleneck = nn.Linear(fc_dim, embed_dim)
        self.feature_extractor = nn.Sequential(self.backbone, self.bottleneck)
        self.fc = nn.Linear(embed_dim, nclass)
        self.domain_classifier = DomainClassifier(embed_dim * nclass, hidden_dim)
        self.nda = embed_dim * nclass

    def forward(self, x, alpha):
        b, _, _, _ = list(map(lambda x: int(x), x.shape))
        feature = self.feature_extractor(x)
        reverse_feature = GRL.apply(feature, alpha)
        class_prediction = self.fc(feature)
        attn_feature = (F.softmax(class_prediction.detach(), dim=-1).unsqueeze(2) @ reverse_feature.unsqueeze(1)).view(
            b, self.nda)
        domain_prediction = self.domain_classifier(attn_feature)
        return feature, class_prediction, domain_prediction

    def predict(self, x):
        feature = self.feature_extractor(x)
        class_prediction = self.fc(feature)
        return class_prediction


def get_model(backbone, fc_dim=2048, embed_dim=256, nclass=31, hidden_dim=1024):
    model = CDAN(backbone, fc_dim=fc_dim, embed_dim=embed_dim, nclass=nclass, hidden_dim=hidden_dim)

    for name, param in model.named_parameters():
        if 'backbone' not in name:
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0)

    return model
