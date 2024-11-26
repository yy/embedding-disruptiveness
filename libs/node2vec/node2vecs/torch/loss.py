# -*- coding: utf-8 -*-
# @Author: Sadamori Kojaku
# @Date:   2022-10-14 14:33:29
# @Last Modified by:   Munjung Kim
# @Last Modified time: 2023-06-03 12:59:28
import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum

# Loss functions
# Node2VecTripletLoss is being used in the disruption project

class DistanceMetrics(Enum):
    """
    The metric for the loasses
    """

    EUCLIDEAN = lambda x, y: F.pairwise_distance(x, y, p=2)
    COSINE = lambda x, y: 1 - F.cosine_similarity(x, y)
    ANGULAR = lambda x, y: torch.arccos((1 - 1e-2) * F.cosine_similarity(x, y))
    DOTSIM = lambda x, y: -(x * y).sum(dim=1)

    def is_scale_invariant(dist_metric):
        return torch.isclose(
            dist_metric(torch.ones(1, 2), torch.ones(1, 2)),
            dist_metric(torch.ones(1, 2), 2 * torch.ones(1, 2)),
        )

    
    
class Node2VecTripletLoss(nn.Module):
    def __init__(self, n_neg):
        super(Node2VecTripletLoss, self).__init__()
        self.n_neg = n_neg
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, model, iwords, owords, nwords):
        ivectors = model.forward(iwords.to('cuda:0')).unsqueeze(2).to('cuda:1') # put target words into cuda:0 and transfer its vectors in cuda:0 into cuda:1
        ovectors = model.forward(owords.to('cuda:1'),forward_in = False)
        nvectors = model.forward(nwords.to('cuda:1'),forward_in=False).neg()
        if self.logsigmoid(torch.bmm(ovectors, ivectors).squeeze()).dim() == 1:
            oloss = self.logsigmoid(torch.bmm(ovectors, ivectors).squeeze()).mean(dim=0)
            
        else:
            oloss = self.logsigmoid(torch.bmm(ovectors, ivectors).squeeze()).mean(dim=1)
       
        nloss = (
            self.logsigmoid(torch.bmm(nvectors, ivectors).squeeze())
            .view(-1, owords.size()[1], self.n_neg)
            .sum(dim=2)
            .mean(dim=1)
        )
    
        
        return -(oloss + nloss).mean()
    
class TripletLoss(nn.Module):
    def __init__(self, model, dist_metric=DistanceMetrics.DOTSIM, with_logsigmoid=True):
        super(TripletLoss, self).__init__()
        self.model = model
        self.weights = None
        self.dist_func = dist_metric
        self.logsigmoid = nn.LogSigmoid()
        self.with_logsigmoid = with_logsigmoid


    def forward(self, iword, oword, y): #4
        ivectors = self.model.forward_i(iword)
        ovectors = self.model.forward_o(oword)

        loss = -self.dist_func(ivectors, ovectors) * y
        if self.with_logsigmoid:
            loss = self.logsigmoid(loss)
        return -(loss).mean()





class ModularityTripletLoss(nn.Module):
    def __init__(self, n_neg):
        super(ModularityTripletLoss, self).__init__()
        self.n_neg = n_neg
        self.logsigmoid = nn.LogSigmoid()

    def forward(self, model, iwords, owords, nwords, base_iwords, base_owords):
        ivectors = model.forward_i(iwords).unsqueeze(2)
        ovectors = model.forward_o(owords)
        nvectors = model.forward_o(nwords).neg()

        base_ivectors = model.forward_i(base_iwords).unsqueeze(2)
        base_ovectors = model.forward_o(base_owords)

        oloss = torch.bmm(ovectors, ivectors).squeeze().mean(dim=1)
        nloss = (
            torch.bmm(nvectors, ivectors)
            .squeeze()
            .view(-1, owords.size()[1], self.n_neg)
            .sum(dim=2)
            .mean(dim=1)
        )

        base_loss = torch.bmm(base_ovectors, base_ivectors).squeeze().mean(dim=1)

        loss = -(oloss + nloss - 0.5 * torch.pow(base_loss, 2)).mean()

        return loss