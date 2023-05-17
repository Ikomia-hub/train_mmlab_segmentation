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
import warnings

from argparse import Namespace
from train_mmlab_segmentation.utils import prepare_dataset, UserStop
import numpy as np
from datetime import datetime

from mmengine.visualization import Visualizer
from mmengine.runner import Runner
from mmengine.config import Config

from mmseg.utils import register_all_modules
import logging

logger = logging.getLogger(__name__)

class MyRunner(Runner):

    @classmethod
    def from_custom_cfg(cls, cfg, custom_hooks, visualizer):
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            optim_wrapper=cfg.get('optim_wrapper'),
            param_scheduler=cfg.get('param_scheduler'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=custom_hooks,
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg'),  # type: ignore
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=visualizer,
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )

        return runner


# --------------------
# - Class to handle the process parameters
# - Inherits PyCore.CWorkflowTaskParam from Ikomia API
# --------------------
class TrainMmlabSegmentationParam(TaskParam):

    def __init__(self):
        TaskParam.__init__(self)
        self.cfg["model_name"] = "segformer"
        self.cfg["model_weight_file"] = "https://download.openmmlab.com/mmsegmentation/v0.5/segformer/" \
                                "segformer_mit-b2_512x512_160k_ade20k/" \
                                "segformer_mit-b2_512x512_160k_ade20k_20210726_112103-cbd414ac.pth"
        self.cfg["model_config"] = "segformer_mit-b2_8xb2-160k_ade20k-512x512"
        self.cfg["max_iter"] = 1000
        self.cfg["batch_size"] = 2
        self.cfg["dataset_split_ratio"] = 0.9
        self.cfg["output_folder"] = os.path.dirname(os.path.realpath(__file__)) + "/runs/"
        self.cfg["eval_period"] = 100
        plugin_folder = os.path.dirname(os.path.realpath(__file__))
        self.cfg["dataset_folder"] = os.path.join(plugin_folder, 'dataset')
        self.cfg["use_custom_model"] = False
        self.cfg["config_file"] = ""

    def set_values(self, param_map):
        self.cfg["model_name"] = param_map["model_name"]
        self.cfg["model_weight_file"] = param_map["model_weight_file"]
        self.cfg["model_config"] = param_map["model_config"]
        self.cfg["max_iter"] = int(param_map["max_iter"])
        self.cfg["batch_size"] = int(param_map["batch_size"])
        self.cfg["dataset_split_ratio"] = float(param_map["dataset_split_ratio"])
        self.cfg["output_folder"] = param_map["output_folder"]
        self.cfg["eval_period"] = int(param_map["eval_period"])
        self.cfg["dataset_folder"] = param_map["dataset_folder"]
        self.cfg["use_custom_model"] = strtobool(param_map["use_custom_model"])
        self.cfg["config_file"] = param_map["config_file"]


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
            self.set_param_object(TrainMmlabSegmentationParam())
        else:
            self.set_param_object(copy.deepcopy(param))

    def get_progress_steps(self, eltCount=1):
        # Function returning the number of progress steps for this process
        # This is handled by the main progress bar of Ikomia application
        return 100

    def update_progress(self):
        self.epochs_done += 1
        steps = range(self.advancement, int(100 * self.iters_done / self.iters_todo))
        for step in steps:
            self.emit_step_progress()
            self.advancement += 1

    def run(self):
        # Core function of your process
        # Call begin_task_run for initialization
        self.begin_task_run()

        self.stop_train = False

        # Get param
        param = self.get_param_object()

        # Get input dataset
        input = self.get_input(0)
        # Current datetime is used as folder name
        str_datetime = datetime.now().strftime("%d-%m-%YT%Hh%Mm%Ss")
        if len(input.data) == 0:
            print("ERROR, there is no input dataset")
            self.end_task_run()
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
                        split_ratio=param.cfg["dataset_split_ratio"], cmap=cmap)

        args = Namespace()
        if os.path.isfile(param.cfg["config_file"]):
            args.config = param.cfg["config_file"]
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
            cfg = Config.fromfile(args.config)
        else:
            if param.cfg["config_file"].startswith("configs"):
                args.config = os.path.join(plugin_folder, param.cfg["config_file"])
            else:
                args.config = os.path.join(plugin_folder, "configs", param.cfg["model_name"],
                                       param.cfg["model_config"] + ".py")
            args.load_from = param.cfg["model_weight_file"] if param.cfg["model_weight_file"] != "" else None
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

            cfg.dataset_type = 'BaseSegDataset'


            classes = [cls for cls in ikdataset["metadata"]["category_names"].values()]

            try:
                cfg.model.decode_head.loss_cls.class_weight = [1.0] * len(classes) + [0.1]
            except:
                pass

            if cmap is not None:
                palette = [list(color) for color in cmap.keys()]
            else:
                palette = np.random.randint(256, size=(len(classes), 3)).tolist()

            for t in cfg.train_pipeline:
                if "reduce_zero_label" in t:
                    t.reduce_zero_label = False
            for t in cfg.test_pipeline:
                if "reduce_zero_label" in t:
                    t.reduce_zero_label = False

            test_dataset = dict(type= cfg.dataset_type,
                        img_suffix = '.jpg',
                        seg_map_suffix = '.png',
                        metainfo = dict(classes = classes, palette= palette),
                        data_root = None,
                        data_prefix= dict(img_path=os.path.join(param.cfg["dataset_folder"], "images", "val"),
                                          seg_map_path=os.path.join(param.cfg["dataset_folder"], "labels", "val")),
                        reduce_zero_label = False,
                        pipeline=cfg.test_pipeline)
            train_dataset = dict(type= cfg.dataset_type,
                        img_suffix='.jpg',
                        seg_map_suffix='.png',
                        metainfo=dict(classes=classes, palette=palette),
                        data_root=None,
                        data_prefix=dict(img_path=os.path.join(param.cfg["dataset_folder"], "images", "train"),
                                         seg_map_path=os.path.join(param.cfg["dataset_folder"], "labels", "train")),
                        reduce_zero_label=False,
                        pipeline=cfg.train_pipeline)

            cfg.train_dataloader = dict(
                batch_size=param.cfg["batch_size"],
                num_workers=0,
                persistent_workers=False,
                sampler=dict(type='DefaultSampler', shuffle=True),
                dataset=train_dataset)

            cfg.test_dataloader = dict(
                batch_size=1,
                num_workers=0,
                persistent_workers=False,
                sampler=dict(type='DefaultSampler', shuffle=True),
                dataset=test_dataset)

            cfg.val_dataloader = cfg.test_dataloader

            if "checkpoint" in cfg.default_hooks:
                cfg.default_hooks.checkpoint["interval"] = -1
                cfg.default_hooks.checkpoint["save_best"] = 'mIoU'
                cfg.default_hooks.checkpoint["rule"] = 'greater'

            cfg.train_cfg = dict(
                type='IterBasedTrainLoop', max_iters=param.cfg["max_iter"], val_interval=param.cfg["eval_period"])

            cfg.param_scheduler = [
                dict(
                    type='LinearLR', start_factor=1e-06, by_epoch=False, begin=0,
                    end=param.cfg["max_iter"]//10),
                dict(
                    type='PolyLR',
                    eta_min=0.0,
                    power=1.0,
                    begin=param.cfg["max_iter"]//10,
                    end=param.cfg["max_iter"],
                    by_epoch=False)
            ]

            cfg.checkpoint_config = dict()
            cfg.model.decode_head.num_classes = len(ikdataset["metadata"]["category_names"])

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
                # load config
                cfg.launcher = args.launcher
                if args.cfg_options is not None:
                    cfg.merge_from_dict(args.cfg_options)

                # work_dir is determined in this priority: CLI > segment in file > filename
                if args.work_dir is not None:
                    # update configs according to CLI args if args.work_dir is not None
                    cfg.work_dir = args.work_dir
                elif cfg.get('work_dir', None) is None:
                    # use config filename as default work_dir if cfg.work_dir is None
                    cfg.work_dir = osp.join('./work_dirs',
                                            osp.splitext(osp.basename(args.config))[0])
        cfg.visualizer = dict(
        type='SegLocalVisualizer',
        vis_backends=[dict(type='TensorboardVisBackend', save_dir=tb_logdir)],
        name='visualizer')

        custom_hooks = [
            dict(type='EmitProgresseAndStopHook', stop=self.get_stop, output_folder=cfg.work_dir,
                 emit_step_progress=self.emit_step_progress, priority='LOWEST'),
            dict(type='CustomLoggerHook', log_metrics=self.log_metrics)
        ]

        register_all_modules(init_default_scope=False)

        try:
            visualizer = Visualizer.get_current_instance()
        except:
            visualizer = cfg.get('visualizer')

        runner = MyRunner.from_custom_cfg(cfg, custom_hooks, visualizer)

        # start training
        try:
            runner.train()
        except UserStop:
            print("Training stopped by user")

        # Step progress bar:
        self.emit_step_progress()

        # Call end_task_run to finalize process
        self.end_task_run()

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
        self.info.short_description = "Train for MMLAB segmentation models"
        self.info.description = "Train for MMLAB segmentation models"
        # relative path -> as displayed in Ikomia application process tree
        self.info.path = "Plugins/Python/Segmentation"
        self.info.icon_path = "icons/mmlab.png"
        self.info.version = "1.1.0"
        # self.info.icon_path = "your path to a specific icon"
        self.info.authors = "MMSegmentation Contributors"
        self.info.article = "{MMSegmentation}: OpenMMLab Semantic Segmentation Toolbox and Benchmark"
        self.info.journal = "publication journal"
        self.info.year = 2021
        self.info.license = "Apache 2.0"
        # URL of documentation
        self.info.documentation_link = "https://mmsegmentation.readthedocs.io/en/latest/"
        # Code source repository
        self.info.repository = "https://github.com/open-mmlab/mmsegmentation"
        # Keywords used for search
        self.info.keywords = "mmlab, train, segmentation"

    def create(self, param=None):
        # Create process object
        return TrainMmlabSegmentation(self.info.name, param)
