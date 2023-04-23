# FLAMNet
PyTorch implementation of the paper "FLAMNet: A Flexible Line Anchor Mechanism Network for Lane Detection". 

## Real traffic scene test demo

## Introduction
![FLAMNet](https://user-images.githubusercontent.com/79684320/233835753-07905d1a-ff30-44ff-9ea8-d68a03030781.png)
- FLAMNet, a lane detection network with a flexible line anchor mechanism, adopts a model architecture where CNN and Transformer are connected in series. 
- FLAMNet achieves SOTA result on CULane, Tusimple, and LLAMAS datasets.


## Installation

### Prerequisites
Only test on Ubuntu18.04 and 20.04 with:
- Python >= 3.8 (tested with Python3.8)
- PyTorch >= 1.6 (tested with Pytorch1.6)
- CUDA (tested with cuda10.2)
- Other dependencies described in `requirements.txt`

### Clone this repository
Clone this code to your workspace. 
We call this directory as `$CLRNET_ROOT`
```Shell
git clone https://github.com/Turoad/clrnet
```

### Create a conda virtual environment and activate it (conda is optional)

```Shell
conda create -n clrnet python=3.8 -y
conda activate clrnet
```

### Install dependencies

```Shell
# Install pytorch firstly, the cudatoolkit version should be same in your system.

conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

# Or you can install via pip
pip install torch==1.8.0 torchvision==0.9.0

# Install python packages
python setup.py build develop
```

### Data preparation

#### CULane

Download [CULane](https://xingangpan.github.io/projects/CULane.html). Then extract them to `$CULANEROOT`. Create link to `data` directory.

```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $CULANEROOT data/CULane
```

For CULane, you should have structure like this:
```
$CULANEROOT/driver_xx_xxframe    # data folders x6
$CULANEROOT/laneseg_label_w16    # lane segmentation labels
$CULANEROOT/list                 # data lists
```


#### Tusimple
Download [Tusimple](https://github.com/TuSimple/tusimple-benchmark/issues/3). Then extract them to `$TUSIMPLEROOT`. Create link to `data` directory.

```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $TUSIMPLEROOT data/tusimple
```

For Tusimple, you should have structure like this:
```
$TUSIMPLEROOT/clips # data folders
$TUSIMPLEROOT/lable_data_xxxx.json # label json file x4
$TUSIMPLEROOT/test_tasks_0627.json # test tasks json file
$TUSIMPLEROOT/test_label.json # test label json file

```

For Tusimple, the segmentation annotation is not provided, hence we need to generate segmentation from the json annotation. 

```Shell
python tools/generate_seg_tusimple.py --root $TUSIMPLEROOT
# this will generate seg_label directory
```

#### LLAMAS
Dowload [LLAMAS](https://unsupervised-llamas.com/llamas/). Then extract them to `$LLAMASROOT`. Create link to `data` directory.

```Shell
cd $CLRNET_ROOT
mkdir -p data
ln -s $LLAMASROOT data/llamas
```

Unzip both files (`color_images.zip` and `labels.zip`) into the same directory (e.g., `data/llamas/`), which will be the dataset's root. For LLAMAS, you should have structure like this:
```
$LLAMASROOT/color_images/train # data folders
$LLAMASROOT/color_images/test # data folders
$LLAMASROOT/color_images/valid # data folders
$LLAMASROOT/labels/train # labels folders
$LLAMASROOT/labels/valid # labels folders
```


## Getting Started

### Training
For training, run
```Shell
python main.py [configs/path_to_your_config] --gpus [gpu_num]
```

For example, run
```Shell
python main.py configs/flamnet/flamnet_dla34_culane.py --gpus 0
```

### Validation
For testing, run
```Shell
python main.py [configs/path_to_your_config] --[test|validate] --load_from [path_to_your_model] --gpus [gpu_num]
```

For example, run
```Shell
python main.py configs/flamnet/flamnet_dla34_culane.py --validate --load_from culane_dla34.pth --gpus 0
```

Currently, this code can output the visualization result when testing, just add `--view`.
We will get the visualization result in `work_dirs/xxx/xxx/visualization`.
