import torch
import torch.nn as nn

from ..registry import LOSSES


def binary_threshold_classification_loss(reg_score,
                                         label,
                                         low_threshold=0.3,
                                         high_threshold=0.7,
                                         ratio_range=(1.05, 21),
                                         eps=1e-5):
    """Binary Threshold Classification Loss."""
    device = reg_score.device
    label = label.view(-1).to(device)
    reg_score = reg_score.contiguous().view(-1)

    hmask = (label >= high_threshold).float().to(device)
    lmask = (label <= low_threshold).float().to(device)
    num_positive = max(torch.sum(hmask), 1)
    num_negative = max(torch.sum(lmask), 1)
    num_entries = num_positive + num_negative

    ratio = num_entries / num_positive
    # clip ratio value between ratio_range
    ratio = min(max(ratio, ratio_range[0]), ratio_range[1])

    coef_n = 0.5 * ratio / (ratio - 1)
    coef_p = 0.5 * ratio
    loss = coef_p * hmask * torch.log(reg_score + eps) + \
        coef_n * lmask * torch.log(1.0 - reg_score + eps)
    loss = -torch.mean(loss)
    return loss


@LOSSES.register_module()
class BinaryThresholdClassificationLoss(nn.Module):
    """Binary Threshold Classification Loss.

    It will calculate binary threshold classification loss given reg_score and
    label at low and high thresholds. Similar to BinaryLogisticRegression loss
    but with tow thresholds.
    """

    def __init__(self,
                 low_threshold=0.3,
                 high_threshold=0.7,
                 ratio_range=(1.05, 21),
                 eps=1e-5):
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.ratio_range = ratio_range
        self.eps = eps

    def forward(self, reg_score, label):
        """Calculate Binary Logistic Regression Loss.

        Args:
                reg_score (torch.Tensor): Predicted score by model.
                label (torch.Tensor): Groundtruth labels.
                # low_threshold (float): Threshold for negative instances.
                #     Default: 0.7.
                # high_threshold (float): Threshold for positive instances.
                #     Default: 0.3.
                # ratio_range (tuple): Lower bound and upper bound for ratio.
                #     Default: (1.05, 21)
                # eps (float): Epsilon for small value. Default: 1e-5.

        Returns:
                torch.Tensor: Returned binary logistic loss.
        """

        return binary_threshold_classification_loss(reg_score, label,
                                                    self.low_threshold,
                                                    self.high_threshold,
                                                    self.ratio_range, self.eps)
