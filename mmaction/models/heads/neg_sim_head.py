import torch.nn as nn
import torch
import torch.distributed as dist

from ..registry import HEADS

@HEADS.register_module
class NegSimHead(nn.Module):
    """Head for RankingLoss.

    """

    def __init__(self,
        ):
        super(NegSimHead, self).__init__()


    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, v_feat, t_feat):
        """Forward head.

        Args:
            v_feat (Tensor): [N * 256]
            t_feat (Tensor): [N * text_num_per_video(T), 256]
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        s = torch.matmul(v_feat, t_feat.permute(1, 0)) # [N , N * T]
        s = s.view(v_feat.shape[0], v_feat.shape[0]) # [N , N]
        N = s.shape[0]
        # Neg Similarity loss
        x = []
        for i in range(N):
            for j in range(N):
                if j != i:
                    x.append(i)

        pos = s[torch.tensor(x), torch.tensor(x)]
        losses = dict()
        losses['neg_sim_loss'] = -torch.mean(pos)

        s = s.view(v_feat.shape[0], v_feat.shape[0], -1)  # [N , N , T]
        with torch.no_grad():
            N = s.shape[0]
            T = s.shape[2]
            s = s.view(v_feat.shape[0], -1)  # [N , N * T]

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
            metric['recall1'] = torch.mean(recall1)
            metric['recall5'] = torch.mean(recall5)
            metric['recall10'] = torch.mean(recall10)
            metric['mean_rk'] = torch.mean(mean_rk)

        return losses, metric
