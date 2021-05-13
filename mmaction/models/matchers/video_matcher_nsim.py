from ..registry import MATCHERS
import torch.nn as nn
from .. import builder
from mmcv.cnn import normal_init
import torch.distributed as dist
import torch
from collections import OrderedDict
from mmcv.runner import auto_fp16


@MATCHERS.register_module()
class VideoMatcherNSim(nn.Module):
    """VideoTextMatcher model framework."""

    def __init__(
        self,
        backbone,
        head,
        neck=None,
        train_cfg=None,
        test_cfg=None,
        fp16_enabled=False,
        img_feat_dim=2048,
        feature_dim=256,
        init_std=0.01,
        gather_flag=True,
        syncBN=False,
        base_momentum=0.996,
    ):
        super().__init__()

        self.backbone1 = builder.build_backbone(backbone)
        self.backbone2 = builder.build_backbone(backbone)
        if neck is not None:
            self.neck = builder.build_neck(neck)
        else:
            self.neck = None
        self.head = builder.build_head(head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.aux_info = []
        if train_cfg is not None and "aux_info" in train_cfg:
            self.aux_info = train_cfg["aux_info"]
        self.fp16_enabled = fp16_enabled
        if fp16_enabled is True:
            self.backbone1 = self.backbone1.half()
            self.backbone2 = self.backbone2.half()
            if neck is not None:
                self.neck = self.neck.half()
            self.head = self.head.half()

        self.img_feat_dim = img_feat_dim
        self.feature_dim = feature_dim
        self.init_std = init_std
        self.syncBN = syncBN

        self.img_mlp1 = nn.Sequential(
            nn.Linear(img_feat_dim, self.feature_dim * 2),
            nn.BatchNorm1d(self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
        )
        self.img_mlp2 = nn.Sequential(
            nn.Linear(img_feat_dim, self.feature_dim * 2),
            nn.BatchNorm1d(self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
        )
        self.predictor_v = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        if syncBN:
            self.img_mlp1 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.img_mlp1)
            self.img_mlp2 = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.img_mlp2)
            self.predictor_v = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
                self.predictor_v
            )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.gather_flag = gather_flag
        self.base_momentum = base_momentum
        self.momentum = base_momentum

        self.init_weights()

    @auto_fp16()
    def extract_feat(self, imgs):
        pass

    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(f"{loss_name} is not a tensor or list of tensors")

        loss = sum(_value for _key, _value in log_vars.items() if "loss" in _key)

        log_vars["loss"] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def init_weights(self):
        self.backbone1.init_weights()
        for layer in self.img_mlp1:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)
        for layer in self.predictor_v:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)

        for backbone_param_ol, backbone_param_tgt in zip(
            self.backbone1.parameters(), self.backbone2.parameters()
        ):
            backbone_param_tgt.data.copy_(backbone_param_ol.data)
        for mlp_param_ol, mlp_param_tgt in zip(
            self.img_mlp1.parameters(), self.img_mlp2.parameters()
        ):
            mlp_param_tgt.data.copy_(mlp_param_ol.data)

    def encoder_v(self, imgs1, imgs2, N):
        x11 = self.backbone1(imgs1)
        if self.avg_pool is not None:
            x11 = self.avg_pool(x11)
        x11 = x11.reshape((N, -1) + x11.shape[1:])
        x11 = x11.mean(dim=1, keepdim=True)
        x11 = x11.squeeze(1)
        # dropout
        x11 = x11.view(x11.size(0), -1)
        p_v1 = self.predictor_v(self.img_mlp1(x11))

        x21 = self.backbone1(imgs2)
        if self.avg_pool is not None:
            x21 = self.avg_pool(x21)
        x21 = x21.reshape((N, -1) + x21.shape[1:])
        x21 = x21.mean(dim=1, keepdim=True)
        x21 = x21.squeeze(1)
        # dropout
        x21 = x21.view(x21.size(0), -1)
        p_v2 = self.predictor_v(self.img_mlp1(x21))

        with torch.no_grad():
            x12 = self.backbone2(imgs1)
            if self.avg_pool is not None:
                x12 = self.avg_pool(x12)
            x12 = x12.reshape((N, -1) + x12.shape[1:])
            x12 = x12.mean(dim=1, keepdim=True)
            x12 = x12.squeeze(1)
            # dropout
            x12 = x12.view(x12.size(0), -1)
            v_feat1 = self.img_mlp2(x12).clone().detach()

            x22 = self.backbone2(imgs2)
            if self.avg_pool is not None:
                x22 = self.avg_pool(x22)
            x22 = x22.reshape((N, -1) + x22.shape[1:])
            x22 = x22.mean(dim=1, keepdim=True)
            x22 = x22.squeeze(1)
            # dropout
            x22 = x22.view(x22.size(0), -1)
            v_feat2 = self.img_mlp2(x22).clone().detach()

        return v_feat1, v_feat2, p_v1, p_v2

    @torch.no_grad()
    def momentum_update(self):
        for backbone_param_ol, backbone_param_tgt in zip(
            self.backbone1.parameters(), self.backbone2.parameters()
        ):
            backbone_param_tgt.data = (
                backbone_param_tgt.data * self.momentum
                + backbone_param_ol.data * (1.0 - self.momentum)
            )

        for mlp_param_ol, mlp_param_tgt in zip(
            self.img_mlp1.parameters(), self.img_mlp2.parameters()
        ):
            mlp_param_tgt.data = (
                mlp_param_tgt.data * self.momentum
                + mlp_param_ol.data * (1.0 - self.momentum)
            )

    def forward(self, imgs1, imgs2, return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(imgs1, imgs2)

        return self.forward_test(imgs1, imgs2)

    def forward_train(self, imgs1, imgs2):
        import pdb

        pdb.set_trace()
        N = imgs1.shape[0]
        imgs1 = imgs1.reshape((-1,) + imgs1.shape[2:])
        imgs2 = imgs2.reshape((-1,) + imgs2.shape[2:])
        v_feat1, v_feat2, p_v1, p_v2 = self.encoder_v(imgs1, imgs2, N)  # [N , C]

        return self.head(v_feat1, v_feat2, p_v1, p_v2)

    def forward_test(self, imgs1, imgs2):
        pass

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
        imgs1 = data_batch["imgs1"]
        imgs2 = data_batch["imgs2"]
        losses, metric = self(imgs1, imgs2)

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
    """Gather tensors from all process, supporting backward propagation.
    """

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
