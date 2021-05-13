import torch.nn as nn
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ..registry import HEADS
import numpy as np


@HEADS.register_module
class NegSimVideoHead(nn.Module):
    """Head for RankingLoss.

    """

    def __init__(self):
        super().__init__()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, v_feat1, v_feat2, p_v1, p_v2):
        """Forward head.

        Args:
            v_feat1 (Tensor): [N , C]
            v_feat2 (Tensor): [N , C]
            p_v1 (Tensor): [N , C]
            p_v2 (Tensor): [N , C]
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        v_feat1 = v_feat1.detach()
        v_feat2 = v_feat2.detach()

        v_feat1 = F.normalize(v_feat1, dim=1)
        v_feat2 = F.normalize(v_feat2, dim=1)
        p_v1 = F.normalize(p_v1, dim=1)
        p_v2 = F.normalize(p_v2, dim=1)

        losses = dict()
        losses["neg_sim_loss"] = (
            -0.5 * (p_v1 * v_feat2).sum(dim=1).mean()
            - 0.5 * (p_v2 * v_feat1).sum(dim=1).mean()
        )

        with torch.no_grad():
            metric = {}
            metric["v_feat1_std"] = torch.mean(torch.std(v_feat1, dim=0))
            metric["v_feat2_std"] = torch.mean(torch.std(v_feat2, dim=0))
            metric["p_v1_std"] = torch.mean(torch.std(p_v1, dim=0))
            metric["p_v2_std"] = torch.mean(torch.std(p_v2, dim=0))

            N = v_feat1.shape[0]
            s1 = torch.matmul(v_feat1, p_v2.permute(1, 0)).view(N, N)  # [N, N]
            s2 = torch.matmul(v_feat2, p_v1.permute(1, 0)).view(N, N)  # [N, N]

            metric = self.retrieval_metric(s1, s2)

        return losses, metric

    def retrieval_metric(self, s1, s2):
        N = s1.shape[0]
        T = 1
        s1 = s1.view(N, -1)  # [N , N * T]

        recall1 = torch.zeros(N).cuda()
        recall5 = torch.zeros(N).cuda()
        recall10 = torch.zeros(N).cuda()
        mean_rk = torch.zeros(N).cuda()

        _, rank = torch.sort(s1, dim=1, descending=True)
        for i in range(N):
            for j in range(N * T):
                if rank[i][j].item() >= T * i and rank[i][j].item() < T * (i + 1):
                    mean_rk[i] += j
                    if j < 10:
                        recall10[i] += 1
                    if j < 5:
                        recall5[i] += 1
                    if j < 1:
                        recall1[i] += 1

        _, rank = torch.sort(s2, dim=1, descending=True)
        for i in range(N):
            for j in range(N * T):
                if rank[i][j].item() >= T * i and rank[i][j].item() < T * (i + 1):
                    mean_rk[i] += j + 1
                    if j < 10:
                        recall10[i] += 1
                    if j < 5:
                        recall5[i] += 1
                    if j < 1:
                        recall1[i] += 1

        recall1 = torch.true_divide(recall1, T)
        recall5 = torch.true_divide(recall5, T)
        recall10 = torch.true_divide(recall10, T)
        mean_rk = torch.true_divide(mean_rk, T)

        metric = dict()
        metric["recall1"] = torch.mean(recall1) / 2
        metric["recall5"] = torch.mean(recall5) / 2
        metric["recall10"] = torch.mean(recall10) / 2
        metric["mean_rk"] = torch.mean(mean_rk) / 2

        return metric
