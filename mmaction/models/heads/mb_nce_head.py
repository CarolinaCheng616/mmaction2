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
            t_feat (Tensor): [N, C]
            v_feat_bank (Tensor): [N, bank_size + 1, C]
            t_feat_bank (Tensor): [N, bank_size + 1, C]
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        device = v_feat.device
        batch_size, feature_dim = v_feat.shape
        bank_size_plus_1 = v_feat_bank.shape[1]
        video_sim = torch.bmm(
            t_feat_bank, v_feat.view(batch_size, feature_dim, 1)
        ).view(
            batch_size, -1
        )  # [batch_size, bank_size + 1]
        video_sim = torch.exp(
            torch.true_divide(video_sim, self.temperature)
        )  # exp^(f(x)g(y)/T)

        text_sim = torch.bmm(v_feat_bank, t_feat.view(batch_size, feature_dim, 1)).view(
            batch_size, -1
        )  # [batch_size, bank_size + 1]
        text_sim = torch.exp(
            torch.true_divide(text_sim, self.temperature)
        )  # exp^(f(x)g(y)/T)

        sim = torch.add(video_sim, text_sim)  # [batch_size, bank_size + 1]
        loss = (
            torch.div(sim[:, 0], torch.sum(sim, dim=1)).log_().mean(0)
        )  # log(exp(fg)/sum(exp(fg)))
        losses = dict()
        losses["memory_bank_nce_loss"] = loss

        metric = dict()
        with torch.no_grad():
            _, rank = torch.sort(
                video_sim, dim=1, descending=True
            )  # [batch_size, bank_size + 1]
            recall1 = torch.zeros(batch_size).to(device)
            recall5 = torch.zeros(batch_size).to(device)
            recall10 = torch.zeros(batch_size).to(device)
            mean_rk = torch.zeros(batch_size).to(device)
            for i in range(batch_size):
                for j in range(bank_size_plus_1):
                    if rank[i][j].item() == 0:
                        mean_rk[i] += j + 1
                        if j < 1:
                            recall1[i] += 1
                        elif j < 5:
                            recall5[i] += 1
                        elif j < 10:
                            recall10[i] += 1
            metric["vt_recall1"] = torch.mean(recall1)
            metric["vt_recall5"] = torch.mean(recall5)
            metric["vt_recall10"] = torch.mean(recall10)
            metric["vt_mean_rk"] = torch.mean(mean_rk)

            _, rank = torch.sort(
                text_sim, dim=1, descending=True
            )  # [batch_size, bank_size + 1]
            recall1 = torch.zeros(batch_size).to(device)
            recall5 = torch.zeros(batch_size).to(device)
            recall10 = torch.zeros(batch_size).to(device)
            mean_rk = torch.zeros(batch_size).to(device)
            for i in range(batch_size):
                for j in range(bank_size_plus_1):
                    if rank[i][j].item() == 0:
                        mean_rk[i] += j + 1
                        if j < 1:
                            recall1[i] += 1
                        elif j < 5:
                            recall5[i] += 1
                        elif j < 10:
                            recall10[i] += 1
            metric["tv_recall1"] = torch.mean(recall1)
            metric["tv_recall5"] = torch.mean(recall5)
            metric["tv_recall10"] = torch.mean(recall10)
            metric["tv_mean_rk"] = torch.mean(mean_rk)
        return losses, metric
