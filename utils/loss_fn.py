# utils/ loss_fn.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2, reduction='mean', eps=1e-8):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps

    def forward(self, inputs, targets):
        inputs = torch.clamp(inputs, self.eps, 1.0 - self.eps)
        BCE_loss = - (targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs))
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class BalancedBCELoss(nn.Module):

    def __init__(self, eps=1e-8):
        super(BalancedBCELoss, self).__init__()
        self.eps = eps

    def forward(self, inputs, targets):
        positive_count = (targets == 1).sum().float()
        negative_count = (targets == 0).sum().float()
        pos_weight = negative_count / (positive_count + self.eps)
        weight_tensor = torch.where(targets == 1, pos_weight, torch.tensor(1.0, device=targets.device))
        loss = F.binary_cross_entropy(inputs, targets, weight=weight_tensor, reduction='mean')
        return loss

class HuberLoss(nn.Module):

    def __init__(self, delta=1.0, reduction='mean'):
        super(HuberLoss, self).__init__()
        self.delta = delta
        self.reduction = reduction

    def forward(self, inputs, targets):
        error = targets - inputs
        abs_error = torch.abs(error)
        is_small_error = abs_error < self.delta
        loss = torch.where(
            is_small_error,
            0.5 * error ** 2,
            self.delta * abs_error - 0.5 * self.delta ** 2
        )
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class LogCoshLoss(nn.Module):

    def __init__(self, reduction='mean'):
        super(LogCoshLoss, self).__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        x = inputs - targets
        loss = torch.log(torch.cosh(x + 1e-12))
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss