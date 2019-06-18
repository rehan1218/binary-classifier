import torch.nn.functional as F
import torch

def nll_loss(output, target):
    target = target.long()
    pred = torch.log(F.softmax(output))
    return F.nll_loss(pred, target)
