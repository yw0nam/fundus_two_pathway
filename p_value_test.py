# %%
import tensorflow as tf
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from  scipy.stats import shapiro, levene
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
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
# dense_non_tilt = proba['non_tilt']['DenseNet121'].mean(0)
# dense_all = proba['All']['DenseNet121'].mean(0)
# dense_tilt = proba['tilt']['DenseNet121'].mean(0)
# # %%
# dense_non_tilt_df = pd.DataFrame(dense_non_tilt, columns=['N','G','P','S'])
# dense_non_tilt_df['label'] = np.argmax(dense_non_tilt, axis=1)
# dense_non_tilt_df['data'] = "non_tilt"
# dense_non_tilt_df['true_label'] = csv_non_tilt['class'].reset_index(drop=True)
# # %%
# dense_tilt_df = pd.DataFrame(dense_tilt, columns=['N','G','P','S'])
# dense_tilt_df['label'] = np.argmax(dense_tilt, axis=1)
# dense_tilt_df['data'] = "tilt"
# dense_tilt_df['true_label'] = csv_tilt['class'].reset_index(drop=True)
# # %%
# dense_all_df = pd.DataFrame(dense_all, columns=['N','G','P','S'])
# dense_all_df['label'] = np.argmax(dense_all, axis=1)
# dense_all_df['data'] = "all"
# dense_all_df['true_label'] = csv['class'].reset_index(drop=True)
# # %%
# df = pd.concat([dense_all_df, dense_non_tilt_df, dense_tilt_df])
# %%
non_tilt_aucs = []
for i in range(5):    
    non_tilt_aucs.append(roc_auc_score(tf.one_hot(csv_non_tilt['class'], 4).numpy(), 
                proba['non_tilt']['DenseNet121'][i, :], multi_class='ovr'))
    
tilt_aucs = []
for i in range(5):    
    tilt_aucs.append(roc_auc_score(tf.one_hot(csv_tilt['class'], 4).numpy(), 
                proba['tilt']['DenseNet121'][i, :], multi_class='ovr'))
    
all_aucs = []
for i in range(5):    
    all_aucs.append(roc_auc_score(tf.one_hot(csv['class'], 4).numpy(), 
                proba['All']['DenseNet121'][i, :], multi_class='ovr'))
# %%
# non_tilt_aucs = []
# for i in range(5):
#     non_tilt_aucs.append(roc_auc_score(tf.one_hot(csv_non_tilt['class'], 4).numpy()[:, 0],
#                                        proba['non_tilt']['DenseNet121'][i, :, 0], multi_class='ovr'))
# # %%
# tilt_aucs = []
# for i in range(5):
#     tilt_aucs.append(roc_auc_score(tf.one_hot(csv_tilt['class'], 4).numpy()[:, 0],
#                                    proba['tilt']['DenseNet121'][i, :, 0], multi_class='ovr'))
# # %%
# all_aucs = []
# for i in range(5):
#     all_aucs.append(roc_auc_score(tf.one_hot(csv['class'], 4).numpy()[:, 0],
#                                   proba['All']['DenseNet121'][i, :, 0], multi_class='ovr'))
# %%
# 정규성을 만족함
print(shapiro(non_tilt_aucs))
print(shapiro(tilt_aucs))
print(shapiro(all_aucs))
# %%
# 등분산성을 유지함
print(levene(non_tilt_aucs, tilt_aucs))
print(levene(non_tilt_aucs, all_aucs))
print(levene(tilt_aucs, all_aucs))
# %%
stats.ttest_ind(non_tilt_aucs, tilt_aucs)
# %%
stats.ttest_ind(non_tilt_aucs, all_aucs)
# %%
stats.ttest_ind(all_aucs, tilt_aucs)
# %%
non_tilt_auc = pd.DataFrame(non_tilt_aucs, columns=['auc'])
non_tilt_auc['data'] = 'non_tilt'

tilt_auc = pd.DataFrame(tilt_aucs, columns=['auc'])
tilt_auc['data'] = 'tilt'

all_auc = pd.DataFrame(all_aucs, columns=['auc'])
all_auc['data'] = 'all'

df = pd.concat([non_tilt_auc, tilt_auc, all_auc])
# %%
model = ols('auc ~ C(data)', df).fit()
anova_lm(model)
# %%
