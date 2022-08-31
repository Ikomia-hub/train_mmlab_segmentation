import os
import numpy as np
import random
import shutil
import cv2
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.hooks import LoggerHook
from mmcv.runner.dist_utils import master_only


class UserStop(Exception):
    pass


# Define custom hook to stop process when user uses stop button and to save last checkpoint

@HOOKS.register_module(force=True)
class EmitProgresseAndStopHook(Hook):
    # Check at each iter if the training must be stopped
    def __init__(self, stop, output_folder, emitStepProgress):
        self.stop = stop
        self.output_folder = output_folder
        self.emitStepProgress = emitStepProgress

    def after_epoch(self, runner):
        self.emitStepProgress()

    def after_train_iter(self, runner):
        # Check if training must be stopped and save last model
        if self.stop():
            runner.save_checkpoint(self.output_folder, "latest.pth", create_symlink=False)
            raise UserStop

    def after_run(self, runner):
        runner.save_checkpoint(self.output_folder, "latest.pth", create_symlink=False)


@HOOKS.register_module(force=True)
class CustomMlflowLoggerHook(LoggerHook):
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
                 interval=10,
                 ignore_last=True,
                 reset_flag=False,
                 by_epoch=False):
        super(CustomMlflowLoggerHook, self).__init__(interval, ignore_last,
                                                     reset_flag, by_epoch)
        self.log_metrics = log_metrics

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        if tags:
            self.log_metrics(tags, step=self.get_iter(runner))


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
    splits_folder = os.path.join(save_dir, 'splits')
    for folder in [images_folder, labels_folder, splits_folder]:
        if not os.path.isdir(folder):
            os.mkdir(folder)

    print("Preparing dataset...")
    train_file = os.path.join(splits_folder, 'train.txt')
    with open(train_file, "w") as f:
        f.write("")
    val_file = os.path.join(splits_folder, 'val.txt')
    with open(val_file, "w") as f:
        f.write("")
    images = ikdata['images']
    n = len(images)
    nb_digits = len(str(n))
    train_idx = random.sample(range(n), int(n * split_ratio))
    for i, sample in enumerate(images):
        _, img_ext = os.path.splitext(sample["filename"])
        _, label_ext = os.path.splitext(sample["semantic_seg_masks_file"])
        if i in train_idx:
            file_to_write = train_file
        else:
            file_to_write = val_file
        id_str = fill_with_zeros(str(i), nb_digits)
        with open(file_to_write, 'a') as f:
            f.write(id_str + '\n')

        if cmap is not None:
            seg_gt = cv2.imread(sample["semantic_seg_masks_file"])[..., ::-1]
            seg_gt = rgb2mask(seg_gt, cmap)
            cv2.imwrite(os.path.join(labels_folder, id_str + label_ext), seg_gt)
        else:
            shutil.copyfile(sample["semantic_seg_masks_file"], os.path.join(labels_folder, id_str + label_ext))
        shutil.copyfile(sample["filename"], os.path.join(images_folder, id_str + img_ext))
    print("Dataset prepared!")


def fill_with_zeros(string, length):
    return '0' * (length - len(string)) + string
