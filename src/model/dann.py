from torch import nn

from src.model.grl import GRL


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

    def forward(self, x, alpha):
        x = GRL.apply(x, alpha)
        x = self.layer(x)
        return x


class DANN(nn.Module):
    def __init__(self, backbone, fc_dim=2048, embed_dim=256, nclass=31, hidden_dim=1024):
        super().__init__()
        self.backbone = backbone
        self.bottleneck = nn.Linear(fc_dim, embed_dim)
        self.feature_extractor = lambda x: self.bottleneck(self.backbone(x))
        self.fc = nn.Linear(embed_dim, nclass)
        self.domain_classifier = DomainClassifier(embed_dim, hidden_dim)

    def forward_impl(self, x, alpha):
        feature = self.feature_extractor(x)
        class_prediction = self.fc(feature)
        domain_prediction = self.domain_classifier(feature, alpha)
        return feature, class_prediction, domain_prediction

    def predict(self, x):
        feature = self.feature_extractor(x)
        class_prediction = self.fc(feature)
        return class_prediction

    def forward(self, *args):
        return self.forward_impl(*args) if self.training else self.predict(*args)


def get_dann(backbone, nclass=31, fc_dim=2048, embed_dim=256, hidden_dim=1024):
    model = DANN(backbone, fc_dim=fc_dim, embed_dim=embed_dim, nclass=nclass, hidden_dim=hidden_dim)

    for name, param in model.named_parameters():
        if 'backbone' not in name:
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0)

    return model
