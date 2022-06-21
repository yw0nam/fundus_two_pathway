# %%
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import scipy.stats as stats
from  scipy.stats import shapiro, levene
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
# %%
ori = pd.read_csv('./data/data_2022_04_11_final.csv')
dev, csv = train_test_split(ori, test_size=0.2,
                             random_state=1004, 
                            stratify=ori['class'])
# %%
with open('./data/test_proba_dual.pckl', 'rb') as f:
    proba = pickle.load(f)
# %%
def p_val_list(y_true, y_pred, label, model):
    
    def cal_mat(csv, y_pred, tilt, label, fold_idx, model):
        if tilt == "tilt":            
            y_true = csv.query('tilt == 1')
        else:
            y_true = csv.query('tilt == 0')
        y_true_label = y_true['class'] == label
        y_pred_label = np.argmax(
            y_pred[tilt][model][fold_idx], axis=1) == label
        conf_mat = metrics.confusion_matrix(y_true_label, y_pred_label)
        acc_num = np.array([conf_mat[0][0] + conf_mat[1][1], 
                            conf_mat[0][1] + conf_mat[1][0]])
        sen_num = np.array([conf_mat[0][0], conf_mat[0][1]])
        spe_num = np.array([conf_mat[1][1], conf_mat[1][0]])
        return [acc_num, sen_num, spe_num]
    
    def cal_p_value(tilt_mats, non_tilt_mats):
        try:
            acc_p_val = stats.chi2_contingency(np.stack([tilt_mats[0], non_tilt_mats[0]], axis=0))[1]
        except:
            acc_p_val = stats.fisher_exact(
                np.stack([tilt_mats[0], non_tilt_mats[0]], axis=0))[1]
        try:
            sen_p_val = stats.chi2_contingency(
                np.stack([tilt_mats[1], non_tilt_mats[1]], axis=0))[1]
        except:
            sen_p_val = stats.fisher_exact(
                np.stack([tilt_mats[1], non_tilt_mats[1]], axis=0))[1]
        try:
            spe_p_val = stats.chi2_contingency(
                np.stack([tilt_mats[2], non_tilt_mats[2]], axis=0))[1]
        except:
            spe_p_val = stats.fisher_exact(
                np.stack([tilt_mats[2], non_tilt_mats[2]], axis=0))[1]
        return [acc_p_val, sen_p_val, spe_p_val]

    p_vals = []
    for fold_idx in range(5):
        tilt_mats = cal_mat(y_true, y_pred, 'tilt', label, fold_idx, model)
        non_tilt_mats = cal_mat(
            y_true, y_pred, 'non_tilt', label, fold_idx, model)
        p_vals.append(cal_p_value(tilt_mats, non_tilt_mats))
    p_vals = np.stack(p_vals, axis=0)
    return p_vals[:, 0], p_vals[:, 1], p_vals[:, 2] # accuracy, sensitivity, specificity respectively


dfs = []
models = ['VGG16', 'VGG19', 'DenseNet121']
for model in models:
    for i in range(4):
        acc_p, sen_p, spe_p = p_val_list(csv, proba, i, model)
        df = pd.DataFrame([acc_p, sen_p, spe_p])
        df['label'] = i
        df['model'] = model
        dfs.append(df)
df = pd.concat(dfs)
# %%
df.to_csv('./temp.csv', index=False)
# %%
df
# %%
spe_p
# %%
def cal_mat(csv, y_pred, tilt, label, fold_idx):
    if tilt == "tilt":            
        y_true = csv.query('tilt == 1')
    else:
        y_true = csv.query('tilt == 0')
    y_true_label = y_true['class'] == label
    y_pred_label = np.argmax(y_pred[tilt]['DenseNet121'][fold_idx], axis=1) == label
    conf_mat = metrics.confusion_matrix(y_true_label, y_pred_label)
    acc_num = np.array([conf_mat[0][0] + conf_mat[1][1], 
                        conf_mat[0][1] + conf_mat[1][0]])
    sen_num = np.array([conf_mat[0][0], conf_mat[0][1]])
    spe_num = np.array([conf_mat[1][1], conf_mat[1][0]])
    return [acc_num, sen_num, spe_num]
# %%
tilt_num = cal_mat(csv, proba, 'tilt', 3, 0)
non_tilt_num = cal_mat(csv, proba, 'non_tilt', 3, 0)
# %%
print(non_tilt_num)
print(tilt_num)
# %%
stats.chi2_contingency(np.stack([tilt_num[0], non_tilt_num[0]], axis=0))
# %%
y_pred_label = np.argmax(proba['tilt']['DenseNet121'][0], axis=1)
metrics.confusion_matrix(csv.query('tilt == 1')['class'] == 3, y_pred_label==3)
# %%
metrics.accuracy_score(csv.query('tilt == 1')['class'] == 3, y_pred_label==3)
# %%
y_pred_label = np.argmax(proba['non_tilt']['DenseNet121'][0], axis=1)
metrics.confusion_matrix(csv.query('tilt == 0')['class']==3, y_pred_label==3)
# %%
metrics.multilabel_confusion_matrix(csv.query('tilt == 0')['class'], y_pred_label)
# %%
