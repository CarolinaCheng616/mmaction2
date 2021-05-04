from ..registry import MATCHERS
from .base import BaseMatcher
import torch.nn as nn
from .. import builder
from mmcv.cnn import normal_init
import torch.distributed as dist
import torch
import torch.nn.functional as F


@MATCHERS.register_module()
class VideoTextMatcherNSimMMT(BaseMatcher):
    """VideoTextMatcher model framework."""

    def __init__(
        self,
        vbackbone,
        tbackbone,
        head,
        neck=None,
        train_cfg=None,
        test_cfg=None,
        fp16_enabled=False,
        img_feat_dim=2048,
        text_feat_dim=768,
        feature_dim=256,
        init_std=0.01,
        update_ratio=0.9,
        use_text_mlp=True,
        gather_flag=True,
    ):
        super(VideoTextMatcherNSimMMT, self).__init__(
            vbackbone, tbackbone, head, neck, train_cfg, test_cfg, fp16_enabled
        )

        self.vbackbone1 = self.backbone1
        self.vbackbone2 = builder.build_backbone(vbackbone)
        self.tbackbone1 = self.backbone2
        self.tbackbone2 = builder.build_backbone(tbackbone)

        self.img_feat_dim = img_feat_dim
        self.text_feat_dim = text_feat_dim
        self.feature_dim = feature_dim
        self.init_std = init_std
        self.update_ratio = update_ratio

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
        self.text_mlp1 = nn.Sequential(
            nn.Linear(text_feat_dim, self.feature_dim * 2),
            nn.BatchNorm1d(self.feature_dim * 2),
            nn.ReLU(),
            nn.Linear(self.feature_dim * 2, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
        )
        self.text_mlp2 = nn.Sequential(
            nn.Linear(text_feat_dim, self.feature_dim * 2),
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
        self.predictor_t = nn.Sequential(
            nn.Linear(self.feature_dim, self.feature_dim),
            nn.BatchNorm1d(self.feature_dim),
            nn.ReLU(),
            nn.Linear(self.feature_dim, self.feature_dim),
        )

        self.init_weights()

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.use_text_mlp = use_text_mlp
        self.gather_flag = gather_flag

    def init_backbone_weights(self):
        self.vbackbone1.init_weights()
        self.vbackbone2.init_weights()
        self.tbackbone1.init_weights()
        self.tbackbone2.init_weights()

    def init_mlp_weights(self):
        """Initialize the model network weights."""
        for layer in self.img_mlp1:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)
        for layer in self.img_mlp2:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)
        for layer in self.text_mlp1:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)
        for layer in self.text_mlp2:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)

    def init_predictor_weights(self):
        """Initialize the model network weights."""
        for layer in self.predictor_v:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)
        for layer in self.predictor_t:
            if isinstance(layer, nn.Linear):
                normal_init(layer, std=self.init_std)

    def init_weights(self):
        self.init_backbone_weights()
        self.init_mlp_weights()
        self.init_predictor_weights()

    def encoder_v(self, imgs, N):
        x1 = self.vbackbone1(imgs)
        if self.avg_pool is not None:
            x1 = self.avg_pool(x1)
        x1 = x1.reshape((N, -1) + x1.shape[1:])
        x1 = x1.mean(dim=1, keepdim=True)
        x1 = x1.squeeze(1)
        # dropout
        x1 = x1.view(x1.size(0), -1)
        x1 = self.img_mlp1(x1)

        x2 = self.vbackbone2(imgs)
        if self.avg_pool is not None:
            x2 = self.avg_pool(x2)
        x2 = x2.reshape((N, -1) + x2.shape[1:])
        x2 = x2.mean(dim=1, keepdim=True)
        x2 = x2.squeeze(1)
        # dropout
        x2 = x2.view(x2.size(0), -1)
        x2 = self.img_mlp2(x2)
        return x1, x2

    def encoder_t(self, texts):
        x1 = self.tbackbone1(texts)
        if self.use_text_mlp:
            x1 = self.text_mlp1(x1)

        x2 = self.tbackbone2(texts)
        if self.use_text_mlp:
            x2 = self.text_mlp2(x2)
        return x1, x2

    def update_encoder(self):
        # update visual backbone
        parameters_1 = dict(self.vbackbone1.named_parameters())
        parameters_2 = self.vbackbone2.named_parameters()
        for name, module in parameters_2:
            module.data = module.data * self.update_ratio + parameters_1[name].data * (
                1 - self.update_ratio
            )
        # update mlp layer
        parameters_1 = dict(self.img_mlp1.named_parameters())
        parameters_2 = self.img_mlp2.named_parameters()
        for name, module in parameters_2:
            module.data = module.data * self.update_ratio + parameters_1[name].data * (
                1 - self.update_ratio
            )
        parameters_1 = dict(self.text_mlp1.named_parameters())
        parameters_2 = self.text_mlp2.named_parameters()
        for name, module in parameters_2:
            module.data = module.data * self.update_ratio + parameters_1[name].data * (
                1 - self.update_ratio
            )

    def forward(self, imgs, texts_item, return_loss=True):
        """Define the computation performed at every call."""
        if return_loss:
            return self.forward_train(imgs, texts_item)

        return self.forward_test(imgs, texts_item)

    def forward_train(self, imgs, texts_item):
        N = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        v_feat1, v_feat2 = self.encoder_v(imgs, N)  # [N , C]
        for key in texts_item:
            texts_item[key] = texts_item[key].reshape((-1,) + texts_item[key].shape[2:])
        t_feat1, t_feat2 = self.encoder_t(texts_item)  # [N * text_num_per_video (T), C]

        if self.gather_flag == True:
            v_feat1 = torch.cat(GatherLayer.apply(v_feat1), dim=0)
            v_feat2 = torch.cat(GatherLayer.apply(v_feat2), dim=0)
            t_feat1 = torch.cat(GatherLayer.apply(t_feat1), dim=0)
            t_feat2 = torch.cat(GatherLayer.apply(t_feat2), dim=0)

        if self.neck is not None:
            v_feat1, t_feat1 = self.neck(v_feat1, t_feat1)
            v_feat2, t_feat2 = self.neck(v_feat2, t_feat2)

        p_v = self.predictor_v(v_feat1)
        p_t = self.predictor_t(t_feat1)

        self.update_encoder()

        return self.head(v_feat2, t_feat2, p_v, p_t)

    def forward_test(self, imgs, texts_item):
        N = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        v_feat1, v_feat2 = self.encoder_v(imgs, N)  # [N , C]
        for key in texts_item:
            texts_item[key] = texts_item[key].reshape((-1,) + texts_item[key].shape[2:])
        t_feat1, t_feat2 = self.encoder_t(texts_item)  # [N * text_num_per_video (T), C]

        if self.gather_flag == True:
            v_feat1 = torch.cat(GatherLayer.apply(v_feat1), dim=0)
            v_feat2 = torch.cat(GatherLayer.apply(v_feat2), dim=0)
            t_feat1 = torch.cat(GatherLayer.apply(t_feat1), dim=0)
            t_feat2 = torch.cat(GatherLayer.apply(t_feat2), dim=0)

        if self.neck is not None:
            v_feat1, t_feat1 = self.neck(v_feat1, t_feat1)
            v_feat2, t_feat2 = self.neck(v_feat2, t_feat2)

        p_v = self.predictor_v(v_feat1)
        p_t = self.predictor_t(t_feat1)

        return zip(
            v_feat2.cpu().numpy(), p_t.view(N, -1, v_feat2.shape[1]).cpu().numpy()
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
        losses, metric = self(imgs, texts_item)

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
