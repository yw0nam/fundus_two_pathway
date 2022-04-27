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
from tableone import TableOne
# %%
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[1], True)


model_names = ['vgg', 'vgg16', 'dense121']
model_plot_names = ['VGG19', 'VGG16', 'DenseNet121']

csv = pd.read_csv('./data/data_2022_04_11_final.csv')
csv['class'] = csv['class'].astype(str)
csv['filename'] = csv['filename'].map(lambda x: x[:-4]+'_cropped.jpg')
# %%
cols = ['class']
# %%
mytable = TableOne(csv, columns=cols, categorical=cols,
                   groupby='tilt', nonnormal='bili')
mytable
# %%
seeds = [1, 11, 111, 1111, 1004]

csv, test = train_test_split(csv, test_size=0.2,
                             random_state=1004,
                             stratify=csv['class'])

csv_tilt = test[test['tilt'] == 1]
csv_not_tilt = test[test['tilt'] == 0]
# %%
model_dicts = []

path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification_single/model_non_tilt/model_'
model_dicts += print_evaluate_metrics(path, data=csv_not_tilt,
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

path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification_single/model_all/model_'
model_dicts += print_evaluate_metrics(path, data=csv_tilt,
                                      dataset_name='tilt',
                                      model_names=model_names,
                                      concat=False,
                                      training_data='all')

path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification_single/model_all/model_'
model_dicts += print_evaluate_metrics(path, data=csv_not_tilt,
                                      dataset_name='non_tilt',
                                      model_names=model_names,
                                      concat=False,
                                      training_data='all')

path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification_single/model_tilt/model_'
model_dicts += print_evaluate_metrics(path, data=csv_tilt,
                                      dataset_name='tilt',
                                      model_names=model_names,
                                      concat=False,
                                      training_data='tilt')

# %%
metric_df_single = pd.DataFrame(data=model_dicts)[
    ['training_data', 'dataset', 'model_name', 'accuracy', 'precision', 'f1_score', 'auc']].sort_values('dataset')

# %%
model_dicts_dual = []
path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_non_tilt/model_'
model_dicts_dual += print_evaluate_metrics(path, data=csv_not_tilt,
                                      dataset_name='non_tilt',
                                      concat=True,
                                      model_names=model_names,
                                      training_data='non_tilt')

path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_all/model_'
model_dicts_dual += print_evaluate_metrics(path, data=test,
                                      dataset_name='all',
                                      model_names=model_names,
                                      concat=True,
                                      training_data='all')

path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_all/model_'
model_dicts_dual += print_evaluate_metrics(path, data=csv_not_tilt,
                                      dataset_name='non_tilt',
                                      model_names=model_names,
                                      concat=True,
                                      training_data='all')

path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_all/model_'
model_dicts_dual += print_evaluate_metrics(path, data=csv_tilt,
                                      dataset_name='tilt',
                                      model_names=model_names,
                                      concat=True,
                                      training_data='all')

path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_tilt/model_'
model_dicts_dual += print_evaluate_metrics(path, data=csv_tilt,
                                      dataset_name='tilt',
                                      model_names=model_names,
                                      concat=True,
                                      training_data='tilt')

# %%
metric_df_dual = pd.DataFrame(data=model_dicts_dual)[
    ['training_data', 'dataset', 'model_name', 'accuracy', 'precision', 'f1_score', 'auc']].sort_values('dataset')

# %%
metric_df_dual.to_csv('./data/test_metric_dual.csv',index=False)
metric_df_single.to_csv('./data/test_metric_single.csv', index=False)

# %%
# pd.read_csv('./data/test_metric_dual.csv')
# # %%
# pd.read_csv('./data/test_metric_single.csv')
# # %%


def make_loss_acc_df(path, model_name, name):
    histories = []
    for i in range(5):
        with open(path+'model_{model_name}_history_{idx}.pckl'.format(model_name=model_name, idx=i), 'rb') as fil:
            histories.append(pickle.load(fil))
    loss_ls = [history['val_loss'] for history in histories]
    acc_ls = [history['val_accuracy'] for history in histories]
    temp = {"loss_mean": np.mean(loss_ls, axis=0),
            "loss_std": np.std(loss_ls, axis=0),
            "acc_mean": np.mean(acc_ls, axis=0),
            "acc_std": np.std(acc_ls, axis=0),
            "epoch": list(range(100))}
    df = pd.DataFrame(temp)
    df['model'] = name
    return df
# %%
model_names = ['VGG16', 'VGG19', 'DenseNet121']
path_names = ['vgg16', 'vgg', 'dense121']
path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_tilt/'

df_ls = []
for i in range(3):
    df_ls.append(make_loss_acc_df(path, path_names[i], model_names[i]))

tilt_df = pd.concat(df_ls, ignore_index=True)
tilt_df['dataset_name'] = 'Tilt'

df_ls = []
path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_non_tilt/'
for i in range(3):
    df_ls.append(make_loss_acc_df(path, path_names[i], model_names[i]))

non_tilt_df = pd.concat(df_ls, ignore_index=True)
non_tilt_df['dataset_name'] = 'Non_Tilt'

df_ls = []
path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_all/'
for i in range(3):
    df_ls.append(make_loss_acc_df(path, path_names[i], model_names[i]))

all_df = pd.concat(df_ls, ignore_index=True)
all_df['dataset_name'] = 'All'
df = pd.concat([all_df, tilt_df, non_tilt_df], ignore_index=True)
# %%
fig = plt.figure(figsize=(20, 5))
xlim = (-1, 100)
ylim = (0, 2)
sns.set_style('white')
area1 = fig.add_subplot(1, 3, 1)
area1.set_title('VGG16')
area1.set(xlim=xlim, ylim=ylim)
area2 = fig.add_subplot(1, 3, 2)
area2.set_title('VGG19')
area2.set(xlim=xlim, ylim=ylim)
area3 = fig.add_subplot(1, 3, 3)
area3.set_title('DenseNet121')
area3.set(xlim=xlim, ylim=ylim)
sns.lineplot(data=df.query("model == 'VGG16'"), x='epoch',
             y='loss_mean', hue='dataset_name', ax=area1)
sns.lineplot(data=df.query("model == 'VGG19'"), x='epoch',
             y='loss_mean', hue='dataset_name', ax=area2)
sns.lineplot(data=df.query("model == 'DenseNet121'"), x='epoch',
             y='loss_mean', hue='dataset_name', ax=area3)
plt.savefig('./figures/Paper/Loss.png')
# %%
fig = plt.figure(figsize=(20, 5))
xlim = (-1, 100)
ylim = (0, 1)
sns.set_style('white')
area1 = fig.add_subplot(1, 3, 1)
area1.set_title('VGG16')
area1.set(xlim=xlim, ylim=ylim)
area2 = fig.add_subplot(1, 3, 2)
area2.set_title('VGG19')
area2.set(xlim=xlim, ylim=ylim)
area3 = fig.add_subplot(1, 3, 3)
area3.set_title('DenseNet121')
area3.set(xlim=xlim, ylim=ylim)
sns.lineplot(data=df.query("model == 'VGG16'"), x='epoch',
             y='acc_mean', hue='dataset_name', ax=area1)
sns.lineplot(data=df.query("model == 'VGG19'"), x='epoch',
             y='acc_mean', hue='dataset_name', ax=area2)
sns.lineplot(data=df.query("model == 'DenseNet121'"), x='epoch',
             y='acc_mean', hue='dataset_name', ax=area3)
plt.savefig('./figures/Paper/Accuracy.png')

# %%
model_names = ['VGG16', 'VGG19', 'DenseNet121']
path_names = ['vgg16', 'vgg', 'dense121']
path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification_single/model_tilt/'

df_ls = []
for i in range(3):
    df_ls.append(make_loss_acc_df(path, path_names[i], model_names[i]))

tilt_df = pd.concat(df_ls, ignore_index=True)
tilt_df['dataset_name'] = 'Tilt'

df_ls = []
path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification_single/model_non_tilt/'
for i in range(3):
    df_ls.append(make_loss_acc_df(path, path_names[i], model_names[i]))

non_tilt_df = pd.concat(df_ls, ignore_index=True)
non_tilt_df['dataset_name'] = 'Non_Tilt'

df_ls = []
path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification_single/model_all/'
for i in range(3):
    df_ls.append(make_loss_acc_df(path, path_names[i], model_names[i]))

all_df = pd.concat(df_ls, ignore_index=True)
all_df['dataset_name'] = 'All'
df = pd.concat([all_df, tilt_df, non_tilt_df], ignore_index=True)
# %%
fig = plt.figure(figsize=(20, 5))
xlim = (-1, 100)
ylim = (0, 2)
sns.set_style('white')
area1 = fig.add_subplot(1, 3, 1)
area1.set_title('VGG16')
area1.set(xlim=xlim, ylim=ylim)
area2 = fig.add_subplot(1, 3, 2)
area2.set_title('VGG19')
area2.set(xlim=xlim, ylim=ylim)
area3 = fig.add_subplot(1, 3, 3)
area3.set_title('DenseNet121')
area3.set(xlim=xlim, ylim=ylim)
sns.lineplot(data=df.query("model == 'VGG16'"), x='epoch',
             y='loss_mean', hue='dataset_name', ax=area1)
sns.lineplot(data=df.query("model == 'VGG19'"), x='epoch',
             y='loss_mean', hue='dataset_name', ax=area2)
sns.lineplot(data=df.query("model == 'DenseNet121'"), x='epoch',
             y='loss_mean', hue='dataset_name', ax=area3)
plt.savefig('./figures/Paper/Loss_single.png')
# %%
fig = plt.figure(figsize=(20, 5))
xlim = (-1, 100)
ylim = (0, 1)
sns.set_style('white')
area1 = fig.add_subplot(1, 3, 1)
area1.set_title('VGG16')
area1.set(xlim=xlim, ylim=ylim)
area2 = fig.add_subplot(1, 3, 2)
area2.set_title('VGG19')
area2.set(xlim=xlim, ylim=ylim)
area3 = fig.add_subplot(1, 3, 3)
area3.set_title('DenseNet121')
area3.set(xlim=xlim, ylim=ylim)
sns.lineplot(data=df.query("model == 'VGG16'"), x='epoch',
             y='acc_mean', hue='dataset_name', ax=area1)
sns.lineplot(data=df.query("model == 'VGG19'"), x='epoch',
             y='acc_mean', hue='dataset_name', ax=area2)
sns.lineplot(data=df.query("model == 'DenseNet121'"), x='epoch',
             y='acc_mean', hue='dataset_name', ax=area3)
plt.savefig('./figures/Paper/Accuracy_single.png')

# %%
