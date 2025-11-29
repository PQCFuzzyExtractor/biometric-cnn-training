# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2


class IrisNet(nn.Module):
    def __init__(self, out_dim=3488):
        super().__init__()

        base = mobilenet_v2(weights="IMAGENET1K_V1")
        base.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.feature_extractor = base.features

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1280, out_dim)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.sigmoid(x)     # 0~1 → 이진화 쉬움
        return x
