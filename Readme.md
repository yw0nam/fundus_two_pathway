# Installation

## Requirements

- Linux
- Python 3.7+
- Tensorflow 2.4.0 or higher and CUDA

a. Create a conda virtual environment and activate it.

```shell
conda create -n fundus python=3.8
conda activate fundus
```

b. Install Tensorflow and CUDA

```shell
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install tensorflow==2.4.1
pip install -q tf-models-official==2.7.0
```

c. Install other libraries

```shell
pip install -r requirements.txt
```
