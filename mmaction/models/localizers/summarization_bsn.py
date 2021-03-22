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
        # x.shape: [N, L, C]
        x = torch.transpose(x, 1, 2)
        # x.shape: [N, C, L]
        x = F.relu(self.conv1_ratio * self.conv1(x))
        x = F.relu(self.conv2_ratio * self.conv2(x))
        x = torch.sigmoid(self.conv3_ratio * self.conv3(x))
        # x.shape: [N, out_dim, L]
        x = x.squeeze(1)
        # x.shape: [N, L]
        return x

    def forward_train(self, features, label_action, video_meta):
        """Define the computation performed at every call when training."""
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
            video_result = np.stack((video_action,), axis=1)  # output: action
            video_results.append((video_name, video_result))
        return video_results

    def forward(self, features, label_action=None, video_meta=None, return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(features, label_action, video_meta)

        return self.forward_test(features, video_meta)


@LOCALIZERS.register_module()
class OriFeatBNPEMReg(BaseLocalizer):
    """Classify proposal into binary category: background or foreground using
    original features.

    Args:
        pem_feat_dim (int): Feature dimension.
        pem_hidden_dim (int): Hidden layer dimension.
        pem_u_ratio_m (float): Ratio for medium score proposals to balance
            data.
        pem_u_ratio_l (float): Ratio for low score proposals to balance data.
        pem_high_temporal_iou_threshold (float): High IoU threshold.
        pem_low_temporal_iou_threshold (float): Low IoU threshold.
        soft_nms_alpha (float): Soft NMS alpha.
        soft_nms_low_threshold (float): Soft NMS low threshold.
        soft_nms_high_threshold (float): Soft NMS high threshold.
        post_process_top_k (int): Top k proposals in post process.
        feature_extraction_interval (int):
            Interval used in feature extraction. Default: 16.
        fc_ratio (float): Ratio for fc layer output. Default: 1.
        classify_ratio (float): Ratio for classify layer output. Default: 1.
        regression_ratio (float): Ratio for regression layer output. Default: 1.  # noqa
        output_dim (int): Output dimension. Default: 1.
        loss_cls (dict): loss class
        classify_loss_ratio (float): Ratio for classify layer output. Default: 1.  # noqa
        regression_loss_ratio (float): Ratio for regression layer output. Default: 1.  # noqa
    """

    def __init__(self,
                 pem_feat_dim,
                 pem_hidden_dim1,
                 pem_hidden_dim2,
                 pem_u_ratio_m,
                 pem_u_ratio_l,
                 pem_high_temporal_iou_threshold,
                 pem_low_temporal_iou_threshold,
                 soft_nms_alpha,
                 soft_nms_low_threshold,
                 soft_nms_high_threshold,
                 post_process_top_k,
                 feature_extraction_interval=16,
                 fc_ratio=1,
                 classify_ratio=1,
                 regression_ratio=1,
                 output_dim=1,
                 loss_cls=dict(type='BinaryThresholdClassificationLoss'),
                 classify_loss_ratio=1,
                 regression_loss_ratio=1,
                 offset_scale=1000):
        super(BaseLocalizer, self).__init__()

        self.feat_dim = pem_feat_dim
        self.hidden_dim1 = pem_hidden_dim1
        self.hidden_dim2 = pem_hidden_dim2
        self.u_ratio_m = pem_u_ratio_m
        self.u_ratio_l = pem_u_ratio_l
        self.pem_high_temporal_iou_threshold = pem_high_temporal_iou_threshold
        self.pem_low_temporal_iou_threshold = pem_low_temporal_iou_threshold
        self.soft_nms_alpha = soft_nms_alpha
        self.soft_nms_low_threshold = soft_nms_low_threshold
        self.soft_nms_high_threshold = soft_nms_high_threshold
        self.post_process_top_k = post_process_top_k
        self.feature_extraction_interval = feature_extraction_interval
        self.fc_ratio = fc_ratio
        self.classify_ratio = classify_ratio
        self.regression_ratio = regression_ratio
        self.output_dim = output_dim
        self.loss_type = loss_cls['type']
        self.loss_cls = build_loss(loss_cls)
        self.regression_loss_ratio = regression_loss_ratio
        self.classify_loss_ratio = classify_loss_ratio
        self.offset_scale = offset_scale

        self.conv = nn.Sequential(
            nn.Conv1d(
                in_channels=self.feat_dim,
                out_channels=self.hidden_dim1,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1), nn.BatchNorm1d(self.hidden_dim1),
            nn.Conv1d(
                in_channels=self.hidden_dim1,
                out_channels=self.hidden_dim1,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=1), nn.BatchNorm1d(self.hidden_dim1))
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(
            in_features=self.hidden_dim1,
            out_features=self.hidden_dim2,
            bias=True)
        self.classify = nn.Linear(
            in_features=self.hidden_dim2,
            out_features=self.output_dim,
            bias=True)
        self.regression = nn.Linear(
            in_features=self.hidden_dim2, out_features=2, bias=True)

    def _forward(self, x):
        """Define the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = torch.cat(list(x))
        # batch, 32, 4096(N, L, C)
        x = torch.transpose(x, 1, 2)
        # transposed to N, C, L
        x = F.relu(self.conv(x))
        # N, C1, L
        x = self.pooling(x).squeeze(2)
        # N, C1
        x = F.relu(self.fc_ratio * self.fc(x))
        # N, C2
        classify = torch.sigmoid(self.classify_ratio * self.classify(x))
        # N, 1
        regression = self.regression_ratio * self.regression(x)
        # N, 2
        return classify, regression

    def forward_train(self, bsp_feature, reference_temporal_iou, offset):
        """Define the computation performed at every call when training."""
        # bsp_feature: list of features, size: videos_per_gpu, feature size
        # e.g. [100*32*4096, 100*32*4096]
        # reference_temporal_iou: list of ious(num:feature num of a video)
        # pem_output: torch.tensor, shape=[videos_per_gpu*feature_size, 1]
        # anchors_temporal_iou.shape=reference_temporal_iou.shape=
        # [videos_per_gpu*feature_size]
        classify, regression = self._forward(bsp_feature)
        reference_temporal_iou = torch.cat(list(reference_temporal_iou))
        offset = torch.cat(list(offset))
        device = classify.device
        reference_temporal_iou = reference_temporal_iou.to(device)
        offset = offset.to(device)
        anchors_temporal_iou = classify.view(-1)
        classify_loss = self.classify_loss_ratio * \
                        self.loss_cls(anchors_temporal_iou, reference_temporal_iou)  # noqa
        positive_idx = reference_temporal_iou >= self.pem_high_temporal_iou_threshold  # noqa
        regression_pos = regression[positive_idx]
        offset_pos = self.offset_scale * offset[positive_idx]
        regression_loss = self.regression_loss_ratio * F.smooth_l1_loss(
            regression_pos, offset_pos)
        loss_dict = dict(
            classify_loss=classify_loss, regression_loss=regression_loss)

        return loss_dict

    def forward_test(self, bsp_feature, tmin, tmax, video_meta):
        """Define the computation performed at every call when testing.

        proposal score is computed by pem_output entirely.
        """
        classify, regression = self._forward(bsp_feature)

        score = classify.view(-1).cpu().numpy().reshape(-1, 1)
        regression = regression.view(-1).cpu().numpy().reshape(-1, 2)
        regression = regression / self.offset_scale

        tmp_tmin = tmin.view(-1).cpu().numpy().reshape(-1, 1)
        tmp_tmax = tmax.view(-1).cpu().numpy().reshape(-1, 1)

        tmin = np.minimum(np.maximum(tmp_tmin + regression[:, 0].reshape(-1, 1), 0), 1)
        tmax = np.minimum(np.maximum(tmp_tmax + regression[:, 1].reshape(-1, 1), 0), 1)

        keep_origin = tmin >= tmax
        tmin[keep_origin] = tmp_tmin[keep_origin]
        tmax[keep_origin] = tmp_tmax[keep_origin]

        result = np.concatenate((tmin, tmax, score), axis=1)
        result = result.reshape(-1, 3)
        video_info = dict(video_meta[0])
        # proposal_list = post_processing_soft_nms(result, video_info,
        #                                          self.soft_nms_alpha,
        #                                          self.soft_nms_low_threshold,
        #                                          self.soft_nms_high_threshold,
        #                                          self.post_process_top_k,
        #                                          self.feature_extraction_interval)
        proposal_list = post_processing_hard_nms(result, video_info,
                                                 self.soft_nms_high_threshold,
                                                 self.post_process_top_k,
                                                 self.feature_extraction_interval)
        output = [
            dict(
                video_name=video_info['video_name'],
                proposal_list=proposal_list)
        ]

        return output

    def forward(self,
                bsp_feature,
                reference_temporal_iou=None,
                tmin=None,
                tmax=None,
                offset=None,
                video_meta=None,
                return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(bsp_feature, reference_temporal_iou,
                                      offset)
        return self.forward_test(bsp_feature, tmin, tmax, video_meta)
