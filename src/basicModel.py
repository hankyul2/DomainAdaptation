from torch import nn


class BasicModel(nn.Module):
    def __init__(self, backbone, fc_dim=2048, embed_dim=256, nclass=31):
        super().__init__()
        self.backbone = backbone
        self.bottleneck = nn.Linear(fc_dim, embed_dim)
        self.feature_extractor = nn.Sequential(self.backbone, self.bottleneck)
        self.fc = nn.Linear(embed_dim, nclass)

    def forward(self, x):
        feature = self.feature_extractor(x)
        class_prediction = self.fc(feature)
        return class_prediction

    def predict(self, x):
        feature = self.feature_extractor(x)
        class_prediction = self.fc(feature)
        return class_prediction


def get_model(backbone, fc_dim=2048, embed_dim=256, nclass=31):
    model = BasicModel(backbone, fc_dim=fc_dim, embed_dim=embed_dim, nclass=nclass)

    for name, param in model.named_parameters():
        if 'backbone' not in name:
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.constant_(param, 0)

    return model