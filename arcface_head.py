# arcface_head.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    """
    ArcFace head
    in_features: embedding dimension (여기서는 3488)
    out_features: 클래스 수 (사람 수)
    """
    def __init__(self, in_features, out_features, s=64.0, m=0.5, easy_margin=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label):
        # input: (B, F), label: (B,)
        # 1) L2 정규화
        x_norm = F.normalize(input)              # (B,F)
        w_norm = F.normalize(self.weight)        # (C,F)

        # 2) cos(theta)
        cosine = F.linear(x_norm, w_norm)        # (B,C)
        sine = torch.sqrt((1.0 - cosine**2).clamp(0, 1))

        # 3) cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # 4) one-hot label
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, label.view(-1, 1), 1.0)

        # 5) ArcFace 적용된 logit
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s

        return output
