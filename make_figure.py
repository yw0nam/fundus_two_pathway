# %%
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score, auc, classification_report, roc_curve
from sklearn.metrics import recall_score, precision_score, RocCurveDisplay
import os, pickle
from tqdm import tqdm
import seaborn as sns
# %%
data = pd.read_csv('./data/data_2022_01_19.csv')


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

# %%
all_df
# %%
df = pd.concat([all_df, tilt_df, non_tilt_df], ignore_index=True)

# %%
fig = plt.figure(figsize=(20, 5))
xlim = (-1, 100)
ylim = (0, 4)
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
sns.set_style('dark')
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
