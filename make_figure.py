# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tables import *
import os, pickle
from tqdm import tqdm
import seaborn as sns
import tensorflow as tf
# %%
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)


model_names = ['vgg', 'vgg16', 'dense121']
model_plot_names = ['VGG19', 'VGG16', 'DenseNet121']

csv = pd.read_csv('./data/data_2022_01_19.csv')
csv['class'] = csv['class'].astype(str)

seeds = [1, 11, 111, 1111, 1004]

csv, test = train_test_split(csv, test_size=0.2,
                             random_state=1004,
                             stratify=csv['class'])

csv_tilt = test[test['tilt'] == 1]
csv_not_tilt = test[test['tilt'] == 0]
# %%
model_dicts = []

path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification_single/model_non_tilt/model_'
model_dicts += print_evaluate_metrics(path, data=test,
                                      dataset_name='non_tilt',
                                      concat=False,
                                      model_names=model_names,
                                      training_data='non_tilt')

path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification_single/model_all/model_'
model_dicts += print_evaluate_metrics(path, data=test,
                                      dataset_name='all',
                                      model_names=model_names,
                                      concat=False,
                                      training_data='all')

path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification_single/model_tilt/model_'
model_dicts += print_evaluate_metrics(path, data=test,
                                      dataset_name='tilt',
                                      model_names=model_names,
                                      concat=False,
                                      training_data='tilt')

# %%
metric_df_single = pd.DataFrame(data=model_dicts)[['dataset', 'model_name', 'accuracy', 'precision', 'f1_score', 'auc']].sort_values('dataset')

# %%
model_dicts_dual = []
path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification_dual/model_non_tilt/model_'
model_dicts_dual += print_evaluate_metrics(path, data=test,
                                      dataset_name='non_tilt',
                                      concat=False,
                                      model_names=model_names,
                                      training_data='non_tilt')

path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification_dual/model_all/model_'
model_dicts_dual += print_evaluate_metrics(path, data=test,
                                      dataset_name='all',
                                      model_names=model_names,
                                      concat=False,
                                      training_data='all')

path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification_dual/model_tilt/model_'
model_dicts_dual += print_evaluate_metrics(path, data=test,
                                      dataset_name='tilt',
                                      model_names=model_names,
                                      concat=False,
                                      training_data='tilt')

# %%
metric_df_dual = pd.DataFrame(data=model_dicts_dual)[
    ['dataset', 'model_name', 'accuracy', 'precision', 'f1_score', 'auc']].sort_values('dataset')

# %%
metric_df_dual.to_csv('./data/test_metric_dual.csv',index=False)
metric_df_single.to_csv('./data/test_metric_single.csv', index=False)

# %%
