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
from ikomia.utils import pyqtutils, qtconversion
from train_mmlab_segmentation.train_mmlab_segmentation_process import TrainMmlabSegmentationParam

# PyQt GUI framework
from PyQt5.QtWidgets import *
import yaml
import os


# --------------------
# - Class which implements widget associated with the process
# - Inherits PyCore.CWorkflowTaskWidget from Ikomia API
# --------------------
class TrainMmlabSegmentationWidget(core.CWorkflowTaskWidget):

    def __init__(self, param, parent):
        core.CWorkflowTaskWidget.__init__(self, parent)

        self.available_cfg_ckpt = None
        if param is None:
            self.parameters = TrainMmlabSegmentationParam()
        else:
            self.parameters = param

        # Create layout : QGridLayout by default
        self.gridLayout = QGridLayout()

        self.combo_model = pyqtutils.append_combo(self.gridLayout, "Model name")

        self.combo_config = pyqtutils.append_combo(self.gridLayout, "Config")

        self.available_models = []
        for dir in os.listdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs")):
            if dir != "_base_":
                self.available_models.append(dir)
                self.combo_model.addItem(dir)
        self.combo_model.currentTextChanged.connect(self.on_model_changed)

        self.combo_model.setCurrentText(self.parameters.cfg["model_name"])

        self.combo_config.setCurrentText(self.parameters.cfg["model_config"])

        self.spin_iters = pyqtutils.append_spin(self.gridLayout, "Iterations", self.parameters.cfg["iters"], min=10)
        self.spin_batch_size = pyqtutils.append_spin(self.gridLayout, "Batch size", self.parameters.cfg["batch_size"])
        self.spin_dataset_percentage = pyqtutils.append_spin(self.gridLayout, "Split train/test (%)",
                                                             self.parameters.cfg["dataset_split_percentage"],
                                                             min=1, max=100)
        self.spin_eval_period = pyqtutils.append_spin(self.gridLayout, "Eval period",
                                                      self.parameters.cfg["eval_period"])
        self.browse_output_folder = pyqtutils.append_browse_file(self.gridLayout, "Output folder",
                                                                 path=self.parameters.cfg["output_folder"],
                                                                 mode=QFileDialog.Directory)
        self.browse_dataset_folder = pyqtutils.append_browse_file(self.gridLayout, "Output dataset folder",
                                                                  path=self.parameters.cfg["dataset_folder"],
                                                                  mode=QFileDialog.Directory)
        self.check_expert_mode = pyqtutils.append_check(self.gridLayout, "Expert mode",
                                                        self.parameters.cfg["expert_mode"])
        self.browse_custom_config = pyqtutils.append_browse_file(self.gridLayout, "Custom config",
                                                                 path=self.parameters.cfg["custom_config"])
        # PyQt -> Qt wrapping
        layout_ptr = qtconversion.PyQtToQt(self.gridLayout)

        # Set widget layout
        self.set_layout(layout_ptr)

    def on_model_changed(self, int):
        self.combo_config.clear()
        model = self.combo_model.currentText()
        yaml_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", model, model + ".yml")
        with open(yaml_file, "r") as f:
            models_list = yaml.load(f, Loader=yaml.FullLoader)['Models']

        self.available_cfg_ckpt = {model_dict["Name"]: {'cfg': model_dict["Config"], 'ckpt': model_dict["Weights"]}
                                   for
                                   model_dict in models_list}
        for experiment_name in self.available_cfg_ckpt.keys():
            self.combo_config.addItem(experiment_name)
        self.combo_config.setCurrentText(list(self.available_cfg_ckpt.keys())[0])

    def on_apply(self):
        # Apply button clicked slot

        # Get parameters from widget
        # Example : self.parameters.windowSize = self.spinWindowSize.value()
        self.parameters.cfg["model_config"] = self.combo_config.currentText()
        self.parameters.cfg["model_name"] = self.combo_model.currentText()
        self.parameters.cfg["model_url"] = self.available_cfg_ckpt[self.parameters.cfg["model_config"]]['ckpt']
        self.parameters.cfg["batch_size"] = self.spin_batch_size.value()
        self.parameters.cfg["iters"] = self.spin_iters.value()
        self.parameters.cfg["eval_period"] = self.spin_eval_period.value()
        self.parameters.cfg["dataset_split_percentage"] = self.spin_dataset_percentage.value()
        self.parameters.cfg["output_folder"] = self.browse_output_folder.path
        self.parameters.cfg["dataset_folder"] = self.browse_dataset_folder.path
        self.parameters.cfg["expert_mode"] = self.check_expert_mode.isChecked()
        self.parameters.cfg["custom_config"] = self.browse_custom_config.path

        # Send signal to launch the process
        self.emit_apply(self.parameters)


# --------------------
# - Factory class to build process widget object
# - Inherits PyDataProcess.CWidgetFactory from Ikomia API
# --------------------
class TrainMmlabSegmentationWidgetFactory(dataprocess.CWidgetFactory):

    def __init__(self):
        dataprocess.CWidgetFactory.__init__(self)
        # Set the name of the process -> it must be the same as the one declared in the process factory class
        self.name = "train_mmlab_segmentation"

    def create(self, param):
        # Create widget object
        return TrainMmlabSegmentationWidget(param, None)
