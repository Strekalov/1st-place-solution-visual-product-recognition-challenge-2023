import torch.nn as nn
import timm
import torch.nn.functional as F


class Trunk(nn.Module):
    def __init__(
        self,
        backbone="efficientnetv2_rw_m",
        pretrained=True,
        embedding_dim=2048,
        dropout=0.0,
    ) -> None:
        super().__init__()

        self.model = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=embedding_dim,
            drop_rate=dropout,
        )

    def forward(self, x):
        x = self.model(x)
        return x


class RetrivealNet(nn.Module):
    def __init__(self, trunk) -> None:
        super().__init__()
        self.trunk = trunk

    def forward(self, x):
        embeddings = self.trunk(x)
        return F.normalize(embeddings, p=2.0, dim=1)
