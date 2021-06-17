from ..registry import RECOGNIZERS
from .base import BaseRecognizer


@RECOGNIZERS.register_module()
class Recognizer2DEpoch(BaseRecognizer):
    """2D recognizer model framework."""

    def train_step(self, data_batch, optimizer, epoch=0, **kwargs):
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
                    epoch (int): current epoch

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
        label = data_batch["label"]

        aux_info = {}
        for item in self.aux_info:
            assert item in data_batch
            aux_info[item] = data_batch[item]

        losses = self(imgs, label, return_loss=True, epoch=epoch, **aux_info)

        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))),
        )

        return outputs

    def forward(self, imgs, label=None, return_loss=True, epoch=0, **kwargs):
        """Define the computation performed at every call."""
        if kwargs.get("gradcam", False):
            del kwargs["gradcam"]
            return self.forward_gradcam(imgs, **kwargs)
        if return_loss:
            if label is None:
                raise ValueError("Label should not be None.")
            return self.forward_train(imgs, label, epoch=epoch, **kwargs)

        return self.forward_test(imgs, **kwargs)

    def forward_train(self, imgs, labels, epoch=0, **kwargs):
        """Defines the computation performed at every call when training."""
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)
        if hasattr(self, "neck"):
            x = [
                each.reshape((-1, num_segs) + each.shape[1:])
                .transpose(1, 2)
                .contiguous()
                for each in x
            ]
            x, _ = self.neck(x, labels.squeeze())
            x = x.squeeze(2)
            num_segs = 1

        cls_score = self.cls_head(x, num_segs)
        gt_labels = labels.squeeze()
        loss_cls = self.cls_head.loss(cls_score, gt_labels, **kwargs)
        losses.update(loss_cls)

        return losses

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]

        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        losses = dict()

        x = self.extract_feat(imgs)
        if hasattr(self, "neck"):
            x = [
                each.reshape((-1, num_segs) + each.shape[1:])
                .transpose(1, 2)
                .contiguous()
                for each in x
            ]
            x, loss_aux = self.neck(x)
            x = x.squeeze(2)
            losses.update(loss_aux)
            num_segs = 1

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop/MultiGroupCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        cls_score = self.cls_head(x, num_segs)

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score, cls_score.size()[0] // batches)

        return cls_score

    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs).cpu().numpy()

    def forward_dummy(self, imgs):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        batches = imgs.shape[0]
        imgs = imgs.reshape((-1,) + imgs.shape[2:])
        num_segs = imgs.shape[0] // batches

        x = self.extract_feat(imgs)
        if hasattr(self, "neck"):
            x = [
                each.reshape((-1, num_segs) + each.shape[1:])
                .transpose(1, 2)
                .contiguous()
                for each in x
            ]
            x, _ = self.neck(x)
            x = x.squeeze(2)
            num_segs = 1

        outs = (self.cls_head(x, num_segs),)
        return outs

    def forward_gradcam(self, imgs):
        """Defines the computation performed at every call when using gradcam
        utils."""
        return self._do_test(imgs)
