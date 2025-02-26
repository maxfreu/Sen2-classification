import torch
import torch.nn.functional as F
from torch.autograd import Variable


# credit to clcarwin
# https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
def focalloss(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        gamma: float,
        reduction=torch.mean):
    """Compute focal loss in multiclass-case

    Args:
        y_pred: Logit Tensor of shape (N, num_classes)
        y_true: Target Tensor of shape (N,)
        gamma: Exponent
        reduction: Function used to reduce (sum / average) the output
    """
    if y_pred.dim() > 2:
        n, c, h, w = y_pred.size()
        y_pred = y_pred.view(n, c, h * w)  # N,C,H,W => N,C,H*W
        y_pred = y_pred.transpose(1, 2)    # N,C,H*W => N,H*W,C
        y_pred = y_pred.contiguous().view(-1, c)   # N,H*W,C => N*H*W,C
    y_true = y_true.view(-1, 1)

    logpt = F.log_softmax(y_pred, dim=-1)
    logpt = logpt.gather(1, y_true)
    logpt = logpt.view(-1)
    pt = logpt.exp()

    loss = -1 * (1-pt)**gamma * logpt
    return reduction(loss)


class FocalLoss(torch.nn.Module):
    def __init__(self, gamma, reduction=torch.mean):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        return focalloss(y_pred, y_true, self.gamma, self.reduction)
