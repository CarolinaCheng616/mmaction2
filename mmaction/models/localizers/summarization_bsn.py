import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import build_loss
from ..registry import LOCALIZERS
from .base import BaseLocalizer


@LOCALIZERS.register_module()
class SumTEM(BaseLocalizer):
    """Temporal Evaluation Model for Boundary Sensetive Network.

    Please refer `BSN: Boundary Sensitive Network for Temporal Action
    Proposal Generation <http://arxiv.org/abs/1806.02964>`_.

    Code reference
    https://github.com/wzmsltw/BSN-boundary-sensitive-network

    Args:
        tem_feat_dim (int): Feature dimension.
        tem_hidden_dim (int): Hidden layer dimension.
        tem_match_threshold (float): Temporal evaluation match threshold.
        loss_cls (dict): Config for building loss.
            Default: ``dict(type='BinaryLogisticRegressionLoss')``.
        loss_weight (float): Weight term for action_loss. Default: 2.
        output_dim (int): Output dimension. Default: 3.
        conv1_ratio (float): Ratio of conv1 layer output. Default: 1.0.
        conv2_ratio (float): Ratio of conv2 layer output. Default: 1.0.
        conv3_ratio (float): Ratio of conv3 layer output. Default: 0.01.
    """

    def __init__(
        self,
        tem_feat_dim,
        tem_hidden_dim,
        tem_match_threshold,
        loss_cls=dict(type="BinaryLogisticRegressionLoss"),
        loss_weight=1,
        output_dim=1,
        conv1_ratio=1,
        conv2_ratio=1,
        conv3_ratio=1,
    ):
        super(BaseLocalizer, self).__init__()

        self.feat_dim = tem_feat_dim
        self.c_hidden = tem_hidden_dim
        self.match_threshold = tem_match_threshold
        self.output_dim = output_dim
        self.loss_cls = build_loss(loss_cls)
        self.loss_weight = loss_weight
        self.conv1_ratio = conv1_ratio
        self.conv2_ratio = conv2_ratio
        self.conv3_ratio = conv3_ratio

        self.conv1 = nn.Conv1d(
            in_channels=self.feat_dim,
            out_channels=self.c_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
        )
        self.conv2 = nn.Conv1d(
            in_channels=self.c_hidden,
            out_channels=self.c_hidden,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=1,
        )
        self.conv3 = nn.Conv1d(
            in_channels=self.c_hidden,
            out_channels=self.output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
        )

    def _forward(self, x):
        """Define the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = F.relu(self.conv1_ratio * self.conv1(x))
        x = F.relu(self.conv2_ratio * self.conv2(x))
        x = torch.sigmoid(self.conv3_ratio * self.conv3(x))
        return x

    def forward_train(self, features, label_action, video_meta):
        """Define the computation performed at every call when training."""
        import pdb

        pdb.set_trace()
        tem_output = self._forward(features)
        score_action = tem_output

        loss_action = self.loss_cls(score_action, label_action, self.match_threshold)
        loss_dict = {"loss_action": loss_action * self.loss_weight}

        return loss_dict

    def forward_test(self, features, video_meta):
        """Define the computation performed at every call when testing."""
        tem_output = self._forward(features).cpu().numpy()
        batch_action = tem_output

        video_meta_list = [dict(x) for x in video_meta]

        video_results = []

        for batch_idx, _ in enumerate(batch_action):
            video_name = video_meta_list[batch_idx]["video_name"]
            video_action = batch_action[batch_idx]
            video_result = np.stack((video_action,), axis=1)
            video_results.append((video_name, video_result))
        return video_results

    def forward(self, features, segments=None, video_meta=None, return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(features, segments, video_meta)

        return self.forward_test(features, video_meta)
