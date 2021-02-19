import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..registry import LOCALIZERS
from .bsn import TEM


@LOCALIZERS.register_module()
class SnippetTEM(TEM):

    def __init__(self, *args, **kwargs):
        super().__init__(
            temporal_dim=2000, boundary_ratio=0.1, *args, **kwargs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(self.output_dim, self.output_dim)

    def _forward(self, x):
        """Define the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = F.relu(self.conv1_ratio * self.conv1(x))
        x = F.relu(self.conv2_ratio * self.conv2(x))
        x = F.relu(self.conv3_ratio * self.conv3(x))
        x = self.pool(x).squeeze(2)
        x = torch.sigmoid(self.linear(x))  # batch, 3
        return x

    def forward_train(self, raw_feature, label_action, label_start, label_end):
        """Define the computation performed at every call when training."""
        # import pdb
        # pdb.set_trace()
        tem_output = self._forward(raw_feature)
        # score_action = tem_output[:, 0, :]
        # score_start = tem_output[:, 1, :]
        # score_end = tem_output[:, 2, :]
        batch = tem_output.shape[0]
        score_action = tem_output[:, 0].reshape(batch, 1)
        score_start = tem_output[:, 1].reshape(batch, 1)
        score_end = tem_output[:, 2].reshape(batch, 1)

        label_action = torch.tensor(label_action)  # [tensor(), tensor(),]
        label_start = torch.tensor(label_start)
        label_end = torch.tensor(label_end)
        label_action = label_action.reshape(label_action.shape[0], 1)
        label_start = label_start.reshape(label_start.shape[0], 1)
        label_end = label_end.reshape(label_end.shape[0], 1)

        loss_action = self.loss_cls(score_action, label_action,
                                    self.match_threshold)
        loss_start_small = self.loss_cls(score_start, label_start,
                                         self.match_threshold)
        loss_end_small = self.loss_cls(score_end, label_end,
                                       self.match_threshold)

        loss_dict = {
            'loss_action': loss_action * self.loss_weight,
            'loss_start': loss_start_small,
            'loss_end': loss_end_small
        }

        return loss_dict

    def forward_test(self, raw_feature, video_meta):
        """Define the computation performed at every call when testing."""
        # import pdb
        # pdb.set_trace()
        tem_output = self._forward(raw_feature).cpu().numpy()
        # batch_action = tem_output[:, 0, :]
        # batch_start = tem_output[:, 1, :]
        # batch_end = tem_output[:, 2, :]

        batch = tem_output.shape[0]
        batch_action = tem_output[:, 0].reshape(batch, 1)
        batch_start = tem_output[:, 1].reshape(batch, 1)
        batch_end = tem_output[:, 2].reshape(batch, 1)

        video_meta_list = [dict(x) for x in video_meta]

        video_results = []

        # for batch_idx, _ in enumerate(batch_action):
        for batch_idx in range(batch):
            video_name = video_meta_list[batch_idx]['video_name']
            video_action = batch_action[batch_idx]
            video_start = batch_start[batch_idx]
            video_end = batch_end[batch_idx]
            duration = int(video_meta_list[batch_idx]['duration_second'])
            idx = int(video_name.split(
                '_')[-1]) + (video_meta_list[batch_idx]['snippet_length']) // 2
            tmin, tmax = np.array([idx / duration
                                   ]), np.array([(idx + 1) / duration])
            video_result = np.stack(
                (video_action, video_start, video_end, tmin, tmax), axis=1)
            video_results.append((video_name, video_result))
        return video_results

    def forward(self,
                raw_feature,
                label_action=None,
                label_start=None,
                label_end=None,
                video_meta=None,
                return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(raw_feature, label_action, label_start,
                                      label_end)

        return self.forward_test(raw_feature, video_meta)


@LOCALIZERS.register_module()
class SnippetTEMSR(TEM):
    # multiclassification: 1 of 4 category
    def __init__(self, *args, **kwargs):
        super().__init__(
            temporal_dim=2000, boundary_ratio=0.1, output_dim=4, loss_cls=dict(type='CrossEntropyLoss'), *args, **kwargs)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(self.output_dim, self.output_dim)

    def _forward(self, x):
        """Define the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The output of the module.
        """
        x = F.relu(self.conv1_ratio * self.conv1(x))
        x = F.relu(self.conv2_ratio * self.conv2(x))
        x = F.relu(self.conv3_ratio * self.conv3(x))
        x = self.pool(x).squeeze(2)
        x = torch.sigmoid(self.linear(x))  # batch, 4
        return x

    def forward_train(self, raw_feature, cate):
        """Define the computation performed at every call when training."""
        import pdb
        pdb.set_trace()
        tem_output = self._forward(raw_feature)
        score = tem_output
        # score = score.to(device=score.device, dtype=)

        # label_action = torch.tensor(label_action)  # [tensor(), tensor(),]
        # label_start = torch.tensor(label_start)
        # label_end = torch.tensor(label_end)
        # label_bg = torch.tensor(label_bg)
        # label_action = label_action.reshape(label_action.shape[0], 1)
        # label_start = label_start.reshape(label_start.shape[0], 1)
        # label_end = label_end.reshape(label_end.shape[0], 1)
        # label_bg = label_bg.reshape(label_bg.shape[0], 1)
        # label = torch.cat((label_action, label_start, label_end, label_bg), dim=1)
        label = torch.tensor(cate)
        # label = label.to(device=score.device, dtype=torch.long)

        # loss_action = self.loss_cls(score_action, label_action,
        #                             self.match_threshold)
        # loss_start_small = self.loss_cls(score_start, label_start,
        #                                  self.match_threshold)
        # loss_end_small = self.loss_cls(score_end, label_end,
        #                                self.match_threshold)

        loss = self.loss_cls(score, label)

        # loss_dict = {
        #     'loss_action': loss_action * self.loss_weight,
        #     'loss_start': loss_start_small,
        #     'loss_end': loss_end_small
        # }
        loss_dict = {
            'loss': loss
        }

        return loss_dict

    def forward_test(self, raw_feature, video_meta):
        """Define the computation performed at every call when testing."""
        # import pdb
        # pdb.set_trace()
        tem_output = self._forward(raw_feature).cpu().numpy()
        # batch_action = tem_output[:, 0, :]
        # batch_start = tem_output[:, 1, :]
        # batch_end = tem_output[:, 2, :]

        batch = tem_output.shape[0]
        batch_action = tem_output[:, 0].reshape(batch, 1)
        batch_start = tem_output[:, 1].reshape(batch, 1)
        batch_end = tem_output[:, 2].reshape(batch, 1)
        batch_bg = tem_output[:, 3].reshape(batch, 1)

        video_meta_list = [dict(x) for x in video_meta]

        video_results = []

        # for batch_idx, _ in enumerate(batch_action):
        for batch_idx in range(batch):
            video_name = video_meta_list[batch_idx]['video_name']
            video_action = batch_action[batch_idx]
            video_start = batch_start[batch_idx]
            video_end = batch_end[batch_idx]
            video_bg = batch_bg[batch_idx]
            duration = int(video_meta_list[batch_idx]['duration_second'])
            idx = int(video_name.split(
                '_')[-1]) + (video_meta_list[batch_idx]['snippet_length']) // 2
            tmin, tmax = np.array([idx / duration
                                   ]), np.array([(idx + 1) / duration])
            video_result = np.stack(
                (video_action, video_start, video_end, video_bg, tmin, tmax), axis=1)
            video_results.append((video_name, video_result))
        return video_results

    def forward(self,
                raw_feature,
                cate=None,
                video_meta=None,
                return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(raw_feature, cate)

        return self.forward_test(raw_feature, video_meta)
