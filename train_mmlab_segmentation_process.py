# Copyright (C) 2021 Ikomia SAS
# Contact: https://www.ikomia.com
#
# This file is part of the IkomiaStudio software.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from ikomia import core, dataprocess
from ikomia.core.task import TaskParam
from ikomia.dnn import dnntrain
from ikomia.core import config as ikcfg
# Your imports below
import os
from distutils.util import strtobool
# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
import torch.distributed as dist
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import Config, DictAction, get_git_hash

from mmseg import __version__
from mmseg.apis import init_random_seed, set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.utils import collect_env, get_root_logger, setup_multi_processes
from argparse import Namespace
from train_mmlab_segmentation.utils import prepare_dataset
import numpy as np
from datetime import datetime

from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import logging

logger = logging.getLogger(__name__)


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainMmlabSegmentationParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        self.cfg["model_name"] = "segformer"
        self.cfg["model_url"] = "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/" \
                                "segformer_mit-b2_512x512_160k_ade20k/" \
                                "segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth"
        self.cfg["model_config"] = "segformer_mit-b2_512x512_160k_ade20k"
        self.cfg["iters"] = 1000
        self.cfg["batch_size"] = 2
        self.cfg["dataset_split_percentage"] = 90
        self.cfg["output_folder"] = os.path.dirname(os.path.realpath(__file__)) + "/runs/"
        self.cfg["eval_period"] = 100
        plugin_folder = os.path.dirname(os.path.realpath(__file__))
        self.cfg["dataset_folder"] = os.path.join(plugin_folder, 'dataset')
        self.cfg["expert_mode"] = False
        self.cfg["custom_config"] = ""

    def setParamMap(self, param_map):
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["model_url"] = param_map["model_url"]
        self.cfg["model_config"] = param_map["model_config"]
        self.cfg["iters"] = int(param_map["iters"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["dataset_split_percentage"] = int(param_map["dataset_split_percentage"])
        self.cfg["output_folder"] = param_map["output_folder"]
        self.cfg["eval_period"] = int(param_map["eval_period"])
        self.cfg["dataset_folder"] = param_map["dataset_folder"]
        self.cfg["expert_mode"] = strtobool(param_map["expert_mode"])
        self.cfg["custom_config"] = param_map["custom_config"]


# --------------------
# - Class which implements the process
# - Inherits PyCore.CWorkflowTask or derived from Ikomia API
# --------------------
class TrainMmlabSegmentation(dnntrain.TrainProcess):

    def __init__(self, name, param):
        dnntrain.TrainProcess.__init__(self, name, param)
        self.stop_train = False
        self.iters_done = None
        self.iters_todo = None
        # Percentage of training done for display purpose
        self.advancement = 0

        # Create parameters class
        if param is None:
            self.setParam(TrainMmlabSegmentationParam())
        else:
            self.setParam(copy.deepcopy(param))

    def getProgressSteps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 100

    def update_progress(self):
        self.epochs_done += 1
        steps = range(self.advancement, int(100 * self.iters_done / self.iters_todo))
        for step in steps:
            self.emitStepProgress()
            self.advancement += 1

    def run(self):
        # Core function of your process
        # Call beginTaskRun for initialization
        self.beginTaskRun()

        self.stop_train = False

        # Get param
        param = self.getParam()

        # Get input dataset
        input = self.getInput(0)
        # Current datetime is used as folder name
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        if len(input.data) == 0:
            print("ERROR, there is no input dataset")
            self.endTaskRun()
            return
        # Tensorboard
        tb_logdir = os.path.join(ikcfg.main_cfg["tensorboard"]["log_uri"], str_datetime)

        ikdataset = input.data
        if "category_colors" in ikdataset["metadata"]:
            cmap = {ikdataset["metadata"]["category_colors"][i]: i for i in
                    range(len(ikdataset["metadata"]["category_colors"]))}
        else:
            cmap = None
        plugin_folder = os.path.dirname(os.path.abspath(__file__))

        prepare_dataset(ikdataset, param.cfg["dataset_folder"],
                        split_ratio=param.cfg["dataset_split_percentage"] / 100, cmap=cmap)

        expert_mode = param.cfg["expert_mode"]

        args = Namespace()
        if expert_mode:
            args.config = param.cfg["custom_config"]
            args.load_from = None
            args.resume_from = None
            args.no_validate = False
            args.gpu_id = 0
            args.seed = None
            args.deterministic = False
            args.cfg_options = None
            args.launcher = 'none'
            args.local_rank = 0
            args.auto_resume = False
            args.gpus = None
            args.gpu_ids = None
            args.diff_seed = False
            args.persistent_workers = True
        else:
            args.config = os.path.join(plugin_folder, "configs", param.cfg["model_name"],
                                       param.cfg["model_config"] + ".py")
            args.load_from = param.cfg["model_url"]
            args.resume_from = None
            args.no_validate = False
            args.gpu_id = 0
            args.seed = None
            args.deterministic = False
            args.cfg_options = None
            args.launcher = 'none'
            args.local_rank = 0
            args.auto_resume = False
            args.gpus = None
            args.gpu_ids = None
            args.diff_seed = False
            args.persistent_workers = True

        args.work_dir = os.path.join(param.cfg["output_folder"], str_datetime)

        cfg = Config.fromfile(args.config)
        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        cfg.dataset_type = 'CustomDataset'
        cfg.data_root = param.cfg["dataset_folder"]
        cfg.data.train.type = cfg.dataset_type
        cfg.data.val.type = cfg.dataset_type
        cfg.data.train.data_root = cfg.data_root
        cfg.data.val.data_root = cfg.data_root
        cfg.data.train.img_dir = 'images'
        cfg.data.val.img_dir = 'images'
        cfg.data.train.ann_dir = 'labels'
        cfg.data.val.ann_dir = 'labels'
        cfg.data.val.classes = [cls for cls in ikdataset["metadata"]["category_names"].values()]
        if cmap is not None:
            cfg.data.val.palette = [list(color) for color in cmap.keys()]
        else:
            cfg.data.val.palette = np.random.randint(256, size=(len(cfg.data.val.classes), 3)).tolist()
        if not expert_mode:
            cfg.optimizer.lr = cfg.optimizer.lr / cfg.data.samples_per_gpu
            cfg.data.samples_per_gpu = param.cfg["batch_size"]
            cfg.data.workers_per_gpu = 1
            cfg.evaluation = dict(interval=param.cfg["eval_period"], metric='mIoU', pre_eval=True, save_best='mIoU',
                                  ignore_index=0)
            cfg.runner = dict(type='IterBasedRunner', max_iters=param.cfg["iters"])

        cfg.data.train.split = "splits/train.txt"
        cfg.data.val.split = "splits/val.txt"
        for elt in cfg.train_pipeline:
            if elt['type'] == 'LoadAnnotations':
                elt['reduce_zero_label'] = False
        for elt in cfg.data.train.pipeline:
            if elt['type'] == 'LoadAnnotations':
                elt['reduce_zero_label'] = False
        cfg.checkpoint_config = dict()
        cfg.model.decode_head.num_classes = len(ikdataset["metadata"]["category_names"])

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        # work_dir is determined in this priority: CLI > segment in file > filename
        if args.work_dir is not None:
            # update configs according to CLI args if args.work_dir is not None
            cfg.work_dir = args.work_dir
        elif cfg.get('work_dir', None) is None:
            # use config filename as default work_dir if cfg.work_dir is None
            cfg.work_dir = osp.join('./work_dirs',
                                    osp.splitext(osp.basename(args.config))[0])
        if args.load_from is not None:
            cfg.load_from = args.load_from

        cfg.model.pretrained = None
        if args.resume_from is not None:
            cfg.resume_from = args.resume_from
        if args.gpus is not None:
            cfg.gpu_ids = range(1)
            warnings.warn('`--gpus` is deprecated because we only support '
                          'single GPU mode in non-distributed training. '
                          'Use `gpus=1` now.')
        if args.gpu_ids is not None:
            cfg.gpu_ids = args.gpu_ids[0:1]
            warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                          'Because we only support single GPU mode in '
                          'non-distributed training. Use the first GPU '
                          'in `gpu_ids` now.')
        if args.gpus is None and args.gpu_ids is None:
            cfg.gpu_ids = [args.gpu_id]

        cfg.auto_resume = args.auto_resume

        # init distributed env first, since logger depends on the dist info.
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)
            # gpu_ids is used to calculate iter when resuming checkpoint
            _, world_size = get_dist_info()
            cfg.gpu_ids = range(world_size)

        # create work_dir
        mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        # init the logger before other steps
        timestamp = str_datetime
        """log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
        """
        # log some basic info
        logger.info(f'Distributed training: {distributed}')
        logger.info(f'Config:\n{cfg.pretty_text}')

        datasets = [build_dataset(cfg.data.train)]
        if len(cfg.workflow) == 2:
            val_dataset = copy.deepcopy(cfg.data.val)
            val_dataset.pipeline = cfg.data.train.pipeline
            datasets.append(build_dataset(val_dataset))

        # save mmseg version, and class names in
        # checkpoints as metadata
        if cfg.checkpoint_config is not None:
            # save mmseg version, config file content and class names in
            # checkpoints as meta data
            cfg.checkpoint_config.meta = dict(
                mmseg_version=f'{__version__}+{get_git_hash()[:7]}',
                # config=cfg.pretty_text,
                CLASSES=cfg.data.val.classes,
                PALETTE=cfg.data.val.palette
            )
        # dump config
        cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

        # set multi-process settings
        setup_multi_processes(cfg)

        # init the meta dict to record some important information such as
        # environment info and seed, which will be logged
        meta = dict()
        # log env info
        env_info_dict = collect_env()
        env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
        dash_line = '-' * 60 + '\n'
        logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                    dash_line)
        meta['env_info'] = env_info

        # set random seeds
        seed = init_random_seed(args.seed)
        seed = seed + dist.get_rank() if args.diff_seed else seed
        logger.info(f'Set random seed to {seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(seed, deterministic=args.deterministic)
        cfg.seed = seed
        meta['seed'] = seed
        meta['exp_name'] = osp.basename(args.config)

        model = build_segmentor(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        model.init_weights()

        # SyncBN is not support for DP
        if not distributed:
            warnings.warn(
                'SyncBN is only supported with DDP. To be compatible with DP, '
                'we convert SyncBN to BN. Please use dist_train.sh which can '
                'avoid this error.')
            model = revert_sync_batchnorm(model)

        logger.info(model)

        # add an attribute for visualization convenience
        model.CLASSES = cfg.data.val.classes
        model.PALETTE = cfg.data.val.palette
        # passing checkpoint meta for saving best checkpoint
        meta.update(cfg.checkpoint_config.meta)

        cfg.custom_hooks = [
            dict(type='EmitProgresseAndStopHook', stop=self.get_stop, output_folder=cfg.work_dir,
                 emitStepProgress=self.emitStepProgress, priority='LOWEST'),
            dict(type='CustomMlflowLoggerHook', log_metrics=self.log_metrics)
        ]
        cfg.log_config = dict(
            interval=10,

            hooks=[
                dict(type='TensorboardLoggerHook', log_dir=tb_logdir)
            ])

        train_segmentor(
            model,
            datasets,
            cfg,
            distributed=distributed,
            validate=(not args.no_validate),
            timestamp=timestamp,
            meta=meta)

        # Step progress bar:
        self.emitStepProgress()

        # Call endTaskRun to finalize process
        self.endTaskRun()

    def get_stop(self):
        return self.stop_train

    def stop(self):
        super().stop()
        self.stop_train = True


# --------------------
# - Factory class to build process object
# - Inherits PyDataProcess.CTaskFactory from Ikomia API
# --------------------
class TrainMmlabSegmentationFactory(dataprocess.CTaskFactory):

    def __init__(self):
        dataprocess.CTaskFactory.__init__(self)
        # Set process information as string here
        self.info.name = "train_mmlab_segmentation"
        self.info.shortDescription = "Train for MMLAB segmentation models"
        self.info.description = "Train for MMLAB segmentation models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.iconPath = "icons/mmlab.png"
        self.info.version = "1.0.0"
        # self.info.iconPath = "your path to a specific icon"
        self.info.authors = "MMSegmentation Contributors"
        self.info.article = "{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark"
        self.info.journal = "publication journal"
        self.info.year = 2021
        self.info.license = "Apache 2.0"
        # URL of documentation
        self.info.documentationLink = "https://mmsegmentation.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/open-mmlab/mmsegmentation"
        # Keywords used for search
        self.info.keywords = "mmlab, train, segmentation"

    def create(self, param=None):
        # Create process object
        return TrainMmlabSegmentation(self.info.name, param)
