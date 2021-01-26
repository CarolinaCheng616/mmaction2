from mmcv.runner import Hook


class ReloadDatasetHook(Hook):
    """Reload dataset before every epoch."""

    def __init__(self):
        pass

    def after_train_epoch(self, runner):
        runner.data_loader.dataset.filter_neg()
