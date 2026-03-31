"""
组合损失函数：Focal Loss + Label Smoothing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    def __init__(self, num_classes=5, alpha=0.25, gamma=2.0, smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def focal_loss(self, inputs, targets):
        """
        Focal Loss for handling class imbalance
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

    def label_smoothing_loss(self, inputs, targets):
        """
        Label Smoothing for preventing overfitting
        """
        log_probs = F.log_softmax(inputs, dim=1)

        # One-hot encoding
        targets_one_hot = torch.zeros_like(inputs)
        targets_one_hot.scatter_(1, targets.unsqueeze(1), 1)

        # Smoothing
        smooth_targets = targets_one_hot * (1 - self.smoothing) + \
            self.smoothing / self.num_classes

        loss = -(smooth_targets * log_probs).sum(dim=1).mean()
        return loss

    def forward(self, inputs, targets):
        """
        Combined loss
        inputs: (B, num_classes)
        targets: (B,)
        """
        fl = self.focal_loss(inputs, targets)
        ls = self.label_smoothing_loss(inputs, targets)

        # Weighted combination
        return 0.6 * fl + 0.4 * ls
