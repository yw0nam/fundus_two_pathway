# optic disc disease classification with tilted


![그림2](https://user-images.githubusercontent.com/39719936/177264029-06df1631-21b2-4aeb-bf1f-a0bbc6ca4ff5.png)


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

# Data distribution

![image](https://user-images.githubusercontent.com/39719936/177264426-0bcb5884-c32c-4139-886d-762e34926ff1.png)

Note, These data collected from Samsung Medical Center.

# Result

![image](https://user-images.githubusercontent.com/39719936/177265243-05c6327e-656b-4b14-8412-15212d5b20a2.png)

In table, Tilted disc shows lower performance compare to Non-Tilted disc
