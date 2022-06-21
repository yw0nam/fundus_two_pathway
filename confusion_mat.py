# %%
from sklearn import metrics
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
# %%
csv = pd.read_csv('./data/data_2022_04_11_final.csv')
_, csv = train_test_split(csv, test_size=0.2,
                          random_state=1004,
                          stratify=csv['class'])

csv_tilt = csv[csv['tilt'] == 1]
csv_non_tilt = csv[csv['tilt'] == 0]
# %%
with open('./data/test_proba_dual.pckl', 'rb') as f:
    proba = pickle.load(f)
# %%
model_names = ["VGG19", 'VGG16', 'DenseNet121']
data_names = ["non_tilt", 'tilt', 'All']
# %%
model_names = ["VGG19", 'VGG16', 'DenseNet121']
data_names = ["non_tilt", 'tilt', 'All']
plot_name = ['non-tilted disc', 'tilted disc', 'All']
use_csvs = [csv_non_tilt, csv_tilt, csv]
for i, data_name in enumerate(data_names):
    use_csv = use_csvs[i]
    for model_name in model_names:
        fig = plt.figure(figsize=(15, 10))
        sns.set(font_scale=1.5)
        sns.set_style('white')
        cm = metrics.confusion_matrix(use_csv['class'], np.argmax(
            np.mean(proba[data_name][model_name], axis=0), axis=1))
        
        sns.heatmap(cm, annot=True, fmt='.3f', cmap='YlGnBu',
                    xticklabels=['N', 'G', 'P', 'S'], yticklabels=['N', 'G', 'P', 'S'], cbar=False)
        plt.title('Data: %s\nModel: %s'%(plot_name[i], model_name), fontsize=25)
        plt.xlabel("Predicted label", fontsize=25)
        plt.ylabel("True label", fontsize=25)
        plt.savefig('./figures/Paper/cm/cm_%s_%s.png'%(data_name, model_name))
# %%
