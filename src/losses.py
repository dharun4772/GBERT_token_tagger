from torch import nn
import torch.nn.functional as F
import torch
from transformers import Trainer
from torch.nn import CrossEntropyLoss
from torch.nn.modules.loss import _Loss
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma=3, weight=None, ignore_index=-100):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ignore_index = ignore_index
    
    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        pt = torch.exp(logpt)
        logpt = (1-pt) ** self.gamma * logpt
        loss = F.nll_loss(logpt, target, weight=self.weight, ignore_index=self.ignore_index)
        return loss


def soft_jaccard_score(output: torch.Tensor, target: torch.Tensor, smooth: float = 0.0, eps: float = 1e-7, dims = None) -> torch.Tensor:
    assert output.size() == target.size(), "Predictions and targets must have the same shape"
    if dims is not None:
        intersection = torch.sum(output * target, dims = dims)
        cardinality = torch.sum(output + target, dims=dims)
    else:
        intersection = torch.sum(output  * target)
        cardinality = torch.sum(output + target)
    
    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return jaccard_score


class JaccardLoss(_Loss):
    def __init__(self, log_loss: bool = False, from_logits: bool = True, smooth: float = 0.0, eps: float = 1e-7):
        super().__init__()
        self.ignore_index = -100
        self.log_loss = log_loss
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
    
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        assert y_pred.size(0) == y_true.size(0), "this would be token length match"

        if self.from_logits:
            y_pred = y_pred.log_softmax(dim=-1).exp()
        
        num_classes = y_pred.size(-1)
        dims = (0)

        y_true = F.one_hot(y_true, num_classes)
        scores = soft_jaccard_score(
            y_pred,
            y_true.type(y_pred.dtype),
            smooth = self.smooth,
            eps = self.eps, 
            dims=dims,
        )

        if self.logg_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1 - scores
        
        mask = y_true.sum(dims) > 0
        loss *= mask.float()

        return loss.mean()
    



