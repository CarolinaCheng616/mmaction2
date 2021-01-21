import torch.nn as nn
import torch.nn.functional as F

from .bsn import TEM


class SnippetTEM(TEM):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
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
        x = self.sigmoid(self.linear(x))  # batch, 3
        return x

    def forward_train(self, raw_feature, label_action, label_start, label_end):
        """Define the computation performed at every call when training."""
        tem_output = self._forward(raw_feature)
        # score_action = tem_output[:, 0, :]
        # score_start = tem_output[:, 1, :]
        # score_end = tem_output[:, 2, :]
        score_action = tem_output[:, 0]
        score_start = tem_output[:, 1]
        score_end = tem_output[:, 2]

        loss_action = self.loss_cls(score_action, label_action,
                                    self.match_threshold)
        loss_start_small = self.loss_cls(score_start, label_start,
                                         self.match_threshold)
        loss_end_small = self.loss_cls(score_end, label_end,
                                       self.match_threshold)
        import pdb
        pdb.set_trace()

        loss_dict = {
            'loss_action': loss_action * self.loss_weight,
            'loss_start': loss_start_small,
            'loss_end': loss_end_small
        }

        return loss_dict

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
