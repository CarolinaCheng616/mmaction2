import torch

# import torch.distributed as dist
import torch.nn as nn

from ..registry import HEADS


@HEADS.register_module
class MBNCEHead(nn.Module):
    """Head for contrastive learning.

    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self, temperature=0.1):
        super(MBNCEHead, self).__init__()
        self.temperature = temperature

    def init_weights(self):
        """Initiate the parameters from scratch."""
        pass

    def forward(self, v_feat, t_feat, v_feat_bank, t_feat_bank):
        """Forward head.

        Args:
            v_feat (Tensor): [N, C]
            t_feat (Tensor): [N * text_num_per_video(T), C]
            v_feat_bank (Tensor): [N, bank_size + 1, C]
            t_feat_bank (Tensor): [N, bank_size + 1, C]
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        batch_size, feature_dim = v_feat.shape
        video_out = torch.bmm(
            t_feat_bank, v_feat.view(batch_size, feature_dim, 1)
        ).view(
            batch_size, -1
        )  # [batch_size, bank_size + 1]
        video_sim = torch.exp(
            torch.true_divide(video_out, self.temperature)
        )  # exp^(f(x)g(y)/T)

        text_out = torch.bmm(v_feat_bank, t_feat.view(batch_size, feature_dim, 1)).view(
            batch_size, -1
        )
        text_sim = torch.exp(
            torch.true_divide(text_out, self.temperature)
        )  # [batch_size, bank_size + 1]
        # exp^(f(x)g(y)/T)

        # sim = torch.cat((video_sim, text_sim), dim=1)  # [batch_size, (bank_size + 1) * 2]
        sim = torch.add(video_sim, text_sim)  # [batch_size, bank_size + 1]
        loss = (
            torch.div(sim[:, 0], torch.sum(sim, dim=1)).log_().mean(0)
        )  # log(exp(fg)/sum(exp(fg)))
        losses = dict()
        losses["memory_bank_nce_loss"] = loss

        metric = None
        # s = torch.matmul(v_feat, t_feat.permute(1, 0))  # [N , N * T]
        # s = torch.true_divide(s, self.temperature)
        # s = s.view(v_feat.shape[0], v_feat.shape[0], -1)  # [N , N , T]

        # MIL-NCE loss
        # nominator = s * torch.eye(s.shape[0])[:, :, None].cuda()
        # nominator = nominator.sum(dim=1)  # [N, T]
        # nominator = torch.logsumexp(nominator, dim=1)  # (N, )
        # denominator = torch.cat((s, s.permute(1, 0, 2)),
        #                         dim=1).view(s.shape[0], -1)
        # denominator = torch.logsumexp(denominator, dim=1)
        #
        # losses = dict()
        # losses['mil_nce_loss'] = torch.mean(denominator - nominator)

        # s = torch.matmul(v_feat, t_feat.permute(1, 0))  # [N , N * T]
        # s = torch.true_divide(s, self.temperature)
        # s = s.view(v_feat.shape[0], v_feat.shape[0], -1)  # [N , N , T]

        # with torch.no_grad():
        #     N = s.shape[0]
        #     T = s.shape[2]
        #     s = s.view(v_feat.shape[0], -1)  # [N , N * T]
        #
        #     _, rank = torch.sort(s, dim=1, descending=True)
        #
        #     recall1 = torch.zeros(N).cuda()
        #     recall5 = torch.zeros(N).cuda()
        #     recall10 = torch.zeros(N).cuda()
        #     mean_rk = torch.zeros(N).cuda()
        #     for i in range(N):
        #         for j in range(N * T):
        #             if T * i <= rank[i][j].item() < T * (i + 1):
        #                 mean_rk[i] += j
        #                 if j < 10:
        #                     recall10[i] += 1
        #                 if j < 5:
        #                     recall5[i] += 1
        #                 if j < 1:
        #                     recall1[i] += 1
        #
        #     recall1 = torch.true_divide(recall1, T)
        #     recall5 = torch.true_divide(recall5, T)
        #     recall10 = torch.true_divide(recall10, T)
        #     mean_rk = torch.true_divide(mean_rk, T)
        #
        #     metric = dict()
        #     metric['recall1'] = torch.mean(recall1)
        #     metric['recall5'] = torch.mean(recall5)
        #     metric['recall10'] = torch.mean(recall10)
        #     metric['mean_rk'] = torch.mean(mean_rk)

        return losses, metric
