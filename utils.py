import os
import numpy as np
import random
import shutil
import cv2
from mmengine.registry import HOOKS
from mmengine.hooks import Hook, LoggerHook


class UserStop(Exception):
    pass


# Define custom hook to stop process when user uses stop button and to save last checkpoint

@HOOKS.register_module(force=True)
class EmitProgresseAndStopHook(Hook):
    # Check at each iter if the training must be stopped
    def __init__(self, stop, output_folder, emit_step_progress):
        self.stop = stop
        self.output_folder = output_folder
        self.emit_step_progress = emit_step_progress

    def after_epoch(self, runner):
        self.emit_step_progress()

    def after_train_iter(self, runner, **kwargs):
        # Check if training must be stopped and save last model
        if self.stop():
            runner.save_checkpoint(self.output_folder, "latest.pth")
            raise UserStop

    def after_run(self, runner, **kwargs):
        runner.save_checkpoint(self.output_folder, "latest.pth")


@HOOKS.register_module(force=True)
class CustomLoggerHook(LoggerHook):
    """Class to log metrics and (optionally) a trained model to MLflow.
    It requires `MLflow`_ to be installed.
    Args:
        interval (int): Logging interval (every k iterations). Default: 10.
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`. Default: True.
        reset_flag (bool): Whether to clear the output buffer after logging.
            Default: False.
        by_epoch (bool): Whether EpochBasedRunner is used. Default: True.
    .. _MLflow:
        https://www.mlflow.org/docs/latest/index.html
    """

    def __init__(self,
                 log_metrics,
                 interval=10):
        super(CustomLoggerHook, self).__init__(interval=interval, log_metric_by_epoch=False)
        self.log_metrics = log_metrics

    def after_val_epoch(self,
                        runner,
                        metrics = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """
        tag, log_str = runner.log_processor.get_log_after_epoch(
            runner, len(runner.val_dataloader), 'val')
        runner.logger.info(log_str)
        if self.log_metric_by_epoch:
            # when `log_metric_by_epoch` is set to True, it's expected
            # that validation metric can be logged by epoch rather than
            # by iter. At the same time, scalars related to time should
            # still be logged by iter to avoid messy visualized result.
            # see details in PR #278.
            metric_tags = {k: v for k, v in tag.items() if 'time' not in k}
            runner.visualizer.add_scalars(
                metric_tags, step=runner.epoch, file_path=self.json_log_path)
            self.log_metrics(tag, step=runner.epoch)
        else:
            runner.visualizer.add_scalars(
                tag, step=runner.iter, file_path=self.json_log_path)
            self.log_metrics(tag, step=runner.iter + 1)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch = None,
                         outputs = None):
        """Record logs after training iteration.

        Args:
            runner (Runner): The runner of the training process.
            batch_idx (int): The index of the current batch in the train loop.
            data_batch (dict tuple or list, optional): Data from dataloader.
            outputs (dict, optional): Outputs from model.
        """
        # Print experiment name every n iterations.
        if self.every_n_train_iters(
                runner, self.interval_exp_name) or (self.end_of_epoch(
                    runner.train_dataloader, batch_idx)):
            exp_info = f'Exp name: {runner.experiment_name}'
            runner.logger.info(exp_info)
        if self.every_n_inner_iters(batch_idx, self.interval):
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        elif (self.end_of_epoch(runner.train_dataloader, batch_idx)
              and not self.ignore_last):
            # `runner.max_iters` may not be divisible by `self.interval`. if
            # `self.ignore_last==True`, the log of remaining iterations will
            # be recorded (Epoch [4][1000/1007], the logs of 998-1007
            # iterations will be recorded).
            tag, log_str = runner.log_processor.get_log_after_iter(
                runner, batch_idx, 'train')
        else:
            return
        runner.logger.info(log_str)
        runner.visualizer.add_scalars(
            tag, step=runner.iter + 1, file_path=self.json_log_path)
        self.log_metrics(tag, step=runner.iter + 1)


def rgb2mask(img, cmap):
    assert len(img.shape) == 3
    height, width, ch = img.shape
    assert ch == 3
    W = np.power(256, [[0], [1], [2]])

    img_id = img.dot(W).squeeze(-1)
    values = np.unique(img_id)
    mask = np.zeros(img_id.shape)

    for c in values:
        try:
            mask[img_id == c] = cmap[tuple(img[img_id == c][0])]
        except:
            pass
    return mask


def prepare_dataset(ikdata, save_dir, split_ratio, cmap):
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    images_folder = os.path.join(save_dir, 'images')
    labels_folder = os.path.join(save_dir, 'labels')
    val_img_folder = os.path.join(images_folder, 'val')
    train_img_folder = os.path.join(images_folder, 'train')
    val_gt_folder = os.path.join(labels_folder, 'val')
    train_gt_folder = os.path.join(labels_folder, 'train')
    shutil.rmtree(images_folder, ignore_errors=True)
    shutil.rmtree(labels_folder, ignore_errors=True)
    for folder in [images_folder, labels_folder, val_img_folder, train_img_folder, val_gt_folder, train_gt_folder]:
        if not os.path.isdir(folder):
            os.mkdir(folder)

    print("Preparing dataset...")
    images = ikdata['images']
    n = len(images)
    nb_digits = len(str(n))
    train_idx = random.sample(range(n), int(n * split_ratio))
    for i, sample in enumerate(images):
        if i in train_idx:
            img_folder = train_img_folder
            gt_folder = train_gt_folder
        else:
            img_folder = val_img_folder
            gt_folder = val_gt_folder
        id_str = fill_with_zeros(str(i), nb_digits)

        if cmap is not None:
            seg_gt = cv2.imread(sample["semantic_seg_masks_file"])[..., ::-1]
            seg_gt = rgb2mask(seg_gt, cmap)
            cv2.imwrite(os.path.join(gt_folder, id_str + '.png'), seg_gt)
        else:
            shutil.copyfile(sample["semantic_seg_masks_file"], os.path.join(gt_folder, id_str + '.png'))
        shutil.copyfile(sample["filename"], os.path.join(img_folder, id_str + '.jpg'))
    print("Dataset prepared!")


def fill_with_zeros(string, length):
    return '0' * (length - len(string)) + string
