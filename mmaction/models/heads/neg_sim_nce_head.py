import torch.nn as nn
import torch
import torch.distributed as dist
import torch.nn.functional as F
from ..registry import HEADS
import numpy as np


@HEADS.register_module
class NegSimNCEHead(nn.Module):
    """Head for RankingLoss.

    """

    def __init__(self, temperature=0.1):
        super(NegSimNCEHead, self).__init__()
        self.temperature = temperature

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, v_feat, t_feat, p_v, p_t):
        """Forward head.

        Args:
            v_feat (Tensor): [N , C]
            t_feat (Tensor): [N , C]
            p_v (Tensor): [N , C]
            p_t (Tensor): [N , C]
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch = v_feat.shape[0]
        v_feat = v_feat.detach()
        t_feat = t_feat.detach()
        v_feat = F.normalize(v_feat, dim=1)
        t_feat = F.normalize(t_feat, dim=1)
        p_v = F.normalize(p_v, dim=1)
        p_t = F.normalize(p_t, dim=1)

        sv = torch.matmul(v_feat, p_t.permute(1, 0))  # [N , N]
        sv = torch.true_divide(sv, self.temperature)

        st = torch.matmul(t_feat, p_v.permute(1, 0))  # [N , N]
        st = torch.true_divide(st, self.temperature)

        # MIL-NCE loss
        nominator_v = (sv * torch.eye(batch)).sum(1).cuda()
        nominator_t = (st * torch.eye(batch)).sum(1).cuda()
        nominator = torch.cat((nominator_v, nominator_t), 0)
        nominator = torch.logsumexp(nominator, dim=0)
        denominator = torch.cat((sv, st), 1)
        denominator = torch.logsumexp(denominator, dim=1)
        losses = dict()
        losses["nsim_mil_nce_loss"] = torch.mean(denominator - nominator)

        with torch.no_grad():
            metric = {}
            metric["v_feat_std"] = torch.mean(torch.std(v_feat, dim=0))
            metric["t_feat_std"] = torch.mean(torch.std(t_feat, dim=0))
            metric["p_v_std"] = torch.mean(torch.std(p_v, dim=0))
            metric["p_t_std"] = torch.mean(torch.std(p_t, dim=0))

            N = v_feat.shape[0]
            sv = torch.matmul(v_feat, p_t.permute(1, 0)).view(N, N)
            st = torch.matmul(t_feat, p_v.permute(1, 0)).view(N, N)

            v_metric = self.retrieval_metric(sv)
            metric["v_recall1"] = v_metric["R1"]
            metric["v_recall5"] = v_metric["R5"]
            metric["v_recall10"] = v_metric["R10"]
            metric["v_mean_rk"] = v_metric["MR"]

            t_metric = self.retrieval_metric(st)
            metric["t_recall1"] = t_metric["R1"]
            metric["t_recall5"] = t_metric["R5"]
            metric["t_recall10"] = t_metric["R10"]
            metric["t_mean_rk"] = t_metric["MR"]

        return losses, metric

    def retrieval_metric(self, s):
        N = s.shape[0]
        T = 1
        s = s.view(N, -1)  # [N , N * T]

        _, rank = torch.sort(s, dim=1, descending=True)

        recall1 = torch.zeros(N).cuda()
        recall5 = torch.zeros(N).cuda()
        recall10 = torch.zeros(N).cuda()
        mean_rk = torch.zeros(N).cuda()
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

        recall1 = torch.true_divide(recall1, T)
        recall5 = torch.true_divide(recall5, T)
        recall10 = torch.true_divide(recall10, T)
        mean_rk = torch.true_divide(mean_rk, T)

        metric = dict()
        metric["R1"] = torch.mean(recall1)
        metric["R5"] = torch.mean(recall5)
        metric["R10"] = torch.mean(recall10)
        metric["MR"] = torch.mean(mean_rk)

        return metric
