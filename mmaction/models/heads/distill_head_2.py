import torch.nn as nn
import torch.nn.functional as F
import torch

from ..registry import HEADS


@HEADS.register_module()
class DistillHead2(nn.Module):
    """
        DistillHead for self-training
    """

    def __init__(self, alpha=0.99, temperature=1, **kwargs):

        super(DistillHead2, self).__init__()
        self.alpha = alpha
        self.temperature = temperature

    def init_weights(self):
        pass

    def forward(self, teacher_cls_score, student_cls_score, labeled, gt_labels):
        """Defines the computation performed at every call.
        Args:
        """
        import pdb

        pdb.set_trace()
        T = self.temperature
        alpha = self.alpha
        losses = dict()

        kl_loss = nn.KLDivLoss()(
            F.log_softmax(student_cls_score / T, dim=1),
            F.softmax(teacher_cls_score / T, dim=1),
        ) * (alpha * T * T)

        # cls_loss = F.cross_entropy(student_cls_score, gt_labels) * (1.0 - alpha)
        cls_loss = F.cross_entropy(student_cls_score[labeled], gt_labels) * (
            1.0 - alpha
        )
        losses["kl_loss"] = torch.mean(kl_loss)
        losses["cls_loss"] = torch.mean(cls_loss)

        return losses
