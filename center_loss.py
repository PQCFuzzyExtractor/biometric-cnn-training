# center_loss.py
import torch
import torch.nn as nn

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.device = device

        # PID 중심 벡터 (class embedding center)
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim).to(device))

    def forward(self, features, labels):
        """
        features: (B, F) - model output embedding
        labels:   (B,)   - PID
        """
        batch_centers = self.centers[labels]          # (B, F)
        loss = ((features - batch_centers) ** 2).sum() / features.size(0)
        return loss
