<div align="center">
  <img src="https://raw.githubusercontent.com/Ikomia-hub/train_mmlab_segmentation/main/icons/mmlab.png" alt="Algorithm icon">
  <h1 align="center">train_mmlab_segmentation</h1>
</div>
<br />
<p align="center">
    <a href="https://github.com/Ikomia-hub/train_mmlab_segmentation">
        <img alt="Stars" src="https://img.shields.io/github/stars/Ikomia-hub/train_mmlab_segmentation">
    </a>
    <a href="https://app.ikomia.ai/hub/">
        <img alt="Website" src="https://img.shields.io/website/http/app.ikomia.ai/en.svg?down_color=red&down_message=offline&up_message=online">
    </a>
    <a href="https://github.com/Ikomia-hub/train_mmlab_segmentation/blob/main/LICENSE.md">
        <img alt="GitHub" src="https://img.shields.io/github/license/Ikomia-hub/train_mmlab_segmentation.svg?color=blue">
    </a>    
    <br>
    <a href="https://discord.com/invite/82Tnw9UGGc">
        <img alt="Discord community" src="https://img.shields.io/badge/Discord-white?style=social&logo=discord">
    </a> 
</p>

Train for MMLAB segmentation models

## :rocket: Use with Ikomia API

#### 1. Install Ikomia API

We strongly recommend using a virtual environment. If you're not sure where to start, we offer a tutorial [here](https://www.ikomia.ai/blog/a-step-by-step-guide-to-creating-virtual-environments-in-python).

```sh
pip install ikomia
```

#### 2. Create your workflow


```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()    

# Add data loader
coco = wf.add_task(name="dataset_coco")

coco.set_parameters({
    "json_file": "path/to/json/annotation/file",
    "image_folder": "path/to/image/folder",
    "task": "semantic_segmentation",
}) 

# Add train algorithm 
train = wf.add_task(name="train_mmlab_segmentation", auto_connect=True)

# Launch your training on your data
wf.run()
```

## :sunny: Use with Ikomia Studio

Ikomia Studio offers a friendly UI with the same features as the API.

- If you haven't started using Ikomia Studio yet, download and install it from [this page](https://www.ikomia.ai/studio).

- For additional guidance on getting started with Ikomia Studio, check out [this blog post](https://www.ikomia.ai/blog/how-to-get-started-with-ikomia-studio).

## :pencil: Set algorithm parameters

- **model_name** (str) - default 'segformer': Name of the model.
- **model_config** (str) - default 'segformer_mit-b2_8xb2-160k_ade20k-512x512': Name of the config.
- **batch_size** (int) - default 2: Number of samples processed before the model is updated. Minimum batch_size is 2.
- **max_iter** (int) - default 1000: Number of training iterations.
- **dataset_split_ratio** (float) – default '0.9': Divide the dataset into train and evaluation sets ]0, 1[.
- **output_folder** (str, *optional*): path to where the model will be saved. 
- **eval_period** (int) - default 100: Number of iterations between 2 evaluations.
- **model_weight_file** (str, *optional*): Model weights used as pretrained model. Will use by default mmlab's weights.
- **config_file** (str, *optional*): Path to the training config file .yaml. Use it only if you know exactly how mmlab works
- **dataset_folder** (str, *optional*): Folder where to save the dataset formatted for mmlab. Is by default in the algorithm directory.

**model_name** and **model_config** work by pair. You can print the available possibilities with this code snippet:

```python
from ikomia.dataprocess.workflow import Workflow

# Init your workflow
wf = Workflow()

# Add algorithm
train = wf.add_task(name="train_mmlab_segmentation")

# Get model zoo and print it
model_zoo = train.get_model_zoo()
print(model_zoo)
```

*Note*: parameter key and value should be in **string format** when added to the dictionary.

```python
...
train.set_parameters({
    "param1": "value1",
    "param2": "value2",
})
...
```