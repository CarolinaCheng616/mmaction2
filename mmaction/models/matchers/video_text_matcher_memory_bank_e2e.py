import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

# from .. import builder
from ..registry import MATCHERS
from .base import BaseMatcher


@MATCHERS.register_module()
class VideoTextMatcherBankE2E(BaseMatcher):
    """VideoTextMatcher model framework."""

    def __init__(
        self,
        backbone1,
        backbone2,
        head,
        neck=None,
        train_cfg=None,
        test_cfg=None,
        fp16_enabled=False,
        dataset_size=200000,
        bank_size=4096,
        img_feat_dim=2048,
        text_feat_dim=768,
        feature_dim=256,
        init_std=0.01,
        bank_update_ratio=0.5,
        use_text_mlp=True,
    ):
        super(VideoTextMatcherBankE2E, self).__init__(
            backbone1, backbone2, head, train_cfg, test_cfg, fp16_enabled
        )
        self.neck = neck
        self.dataset_size = dataset_size
        self.bank_size = bank_size
        stdv = 1.0 / np.sqrt(feature_dim / 3)
        self.register_buffer(
            "video_bank",
            torch.rand(dataset_size, feature_dim).mul_(2 * stdv).add_(-stdv).detach(),
        )  # [-stdv, stdv]
        self.register_buffer(
            "text_bank",
            torch.rand(dataset_size, feature_dim).mul_(2 * stdv).add_(-stdv).detach(),
        )  # [-stdv, stdv]
        self.probs = torch.ones(dataset_size).detach()
        self.img_feat_dim = img_feat_dim
        self.text_feat_dim = text_feat_dim
        self.feature_dim = feature_dim
        self.init_std = init_std
        self.bank_update_ratio = bank_update_ratio

        self.img_mlp = nn.Sequential(
            nn.Linear(img_feat_dim, self.feature_dim * 2),
            nn.BatchNorm1d(self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
        )
        self.text_mlp = nn.Sequential(
            nn.Linear(text_feat_dim, self.feature_dim * 2),
            nn.BatchNorm1d(self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.init_mlp_weights()
        self.use_text_mlp = use_text_mlp

    def init_mlp_weights(self):
        """Initialize the model network weights."""
        for layer in self.img_mlp:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)
        for layer in self.text_mlp:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)

    def encoder_v(self, imgs, N):
        x = self.backbone1(imgs)
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        x = x.reshape((N, -1) + x.shape[1:])
        x = x.mean(dim=1, keepdim=True)
        x = x.squeeze(1)
        # dropout
        x = x.view(x.size(0), -1)
        x = self.img_mlp(x)
        return x

    def encoder_t(self, texts):
        x = self.backbone2(texts)
        if self.use_text_mlp:
            x = self.text_mlp(x)
        return x

    def update_bank(self, v_feat, t_feat, idx):
        # gather
        v_feat = torch.cat(GatherLayer.apply(v_feat), dim=0)
        t_feat = torch.cat(GatherLayer.apply(t_feat), dim=0)
        idx = torch.cat(GatherLayer.apply(idx), dim=0).view(-1)
        with torch.no_grad():
            v_feat_bank = torch.index_select(self.video_bank, 0, idx).detach()
            t_feat_bank = torch.index_select(self.text_bank, 0, idx).detach()
            v_feat_bank.mul_(self.bank_update_ratio).add_(
                torch.mul(v_feat, 1 - self.bank_update_ratio)
            )
            t_feat_bank.mul_(self.bank_update_ratio).add_(
                torch.mul(t_feat, 1 - self.bank_update_ratio)
            )
            v_feat_bank = F.normalize(v_feat_bank, dim=1)
            t_feat_bank = F.normalize(t_feat_bank, dim=1)
            self.video_bank.index_copy_(0, idx, v_feat_bank)
            self.text_bank.index_copy_(0, idx, t_feat_bank)

    def forward(self, imgs, texts_item, idx=None, return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(imgs, texts_item, idx)

        return self.forward_test(imgs, texts_item)

    def forward_train(self, imgs, texts_item, idxes):
        # BNCHW
        device = imgs.device
        batch_size = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        v_feat = F.normalize(self.encoder_v(imgs, batch_size), dim=1)  # [N , C]
        for key in texts_item:
            texts_item[key] = texts_item[key].reshape((-1,) + texts_item[key].shape[2:])
        t_feat = F.normalize(
            self.encoder_t(texts_item), dim=1
        )  # [N * text_num_per_video (T), C] noqa

        if self.neck is not None:
            v_feat, t_feat = self.neck(v_feat, t_feat)

        slct_idx = torch.zeros(
            batch_size, self.bank_size + 1
        ).detach()  # [batch_size, bank_size + 1]
        for i, idx in enumerate(idxes):
            self.probs[idx] = 0.0
            slct_idx[i] = torch.multinomial(
                self.probs, self.bank_size + 1, replacement=True
            ).detach()
            self.probs[idx] = 1.0
        v_feat_bank = (
            torch.index_select(self.video_bank, 0, slct_idx.view(-1))
            .detach()
            .view(batch_size, self.bank_size + 1, self.feature_dim)
        )  # [batch_size, bank_size+1, feature_dim]

        t_feat_bank = (
            torch.index_select(self.text_bank, 0, slct_idx.view(-1))
            .detach()
            .view(batch_size, self.bank_size + 1, self.feature_dim)
        )  # [batch_size, bank_size+1, feature_dim]

        self.update_bank(v_feat, t_feat, idxes)

        return self.head(v_feat, t_feat, v_feat_bank, t_feat_bank)

    def forward_test(self, imgs, texts_item):
        batch_size = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        v_feat = F.normalize(self.encoder_v(imgs, batch_size), dim=1)  # [N , C]
        for key in texts_item:
            texts_item[key] = texts_item[key].reshape((-1,) + texts_item[key].shape[2:])
        t_feat = F.normalize(
            self.encoder_t(texts_item), dim=1
        )  # [N * text_num_per_video (T), C]
        t_feat = t_feat.view(batch_size, -1)  # [N , T * C]

        v_feat = torch.cat(GatherLayer.apply(v_feat), dim=0)
        t_feat = torch.cat(GatherLayer.apply(t_feat), dim=0)

        if self.neck is not None:
            v_feat, t_feat = self.neck(v_feat, t_feat)

        return zip(
            v_feat.cpu().numpy(),
            t_feat.view(batch_size, -1, v_feat.shape[1]).cpu().numpy(),
        )

    def forward_gradcam(self, audios):
        raise NotImplementedError

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        imgs = data_batch["imgs"]
        texts_item = data_batch["texts_item"]
        idxes = data_batch["idxes"]
        losses, metric = self(imgs, texts_item, idxes)

        loss, log_vars = self._parse_losses(losses)

        for key, value in metric.items():
            if dist.is_available() and dist.is_initialized():
                value = value.data.clone()
                dist.all_reduce(value.div_(dist.get_world_size()))
            log_vars[key] = value.item()

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))),
        )

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        pass


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
