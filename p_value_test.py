# %%
from sklearn import metrics
import tensorflow as tf
import pandas as pd
import scipy.stats as stats
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from  scipy.stats import shapiro, levene
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
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

class_names = ['Normal', "Glaucoma",
               "Optic disc pallor", "Optic disc swelling"]

# %%

def cal_metrics_for_each_class(proba, model_name, data_name, csv):
    class_names = ['Normal', "Glaucoma",
                   "Optic disc pallor", "Optic disc swelling"]
    dict_ls = []
    for i in range(4):
        sen_ls = []  # Sensitivity
        spe_ls = []  # Specificity
        pre_ls = []  # Precision
        f1_ls = []  # F1-score
        acc_ls = [] # accuracy 
        auc_ls = [] # AUC
        for j in range(5):
            y_true = csv['class'] == i
            y_pred = np.argmax(proba[data_name][model_name][j], axis=1) == i
            prec, recall, f1, _ = metrics.precision_recall_fscore_support(y_true, y_pred, pos_label=True, average=None)
            acc_ls.append(metrics.accuracy_score(y_true, y_pred))
            auc_ls.append(metrics.roc_auc_score(y_true, proba[data_name][model_name][j, :, i]))
            sen_ls.append(recall[0])
            spe_ls.append(recall[1])
            pre_ls.append(prec[0])
            f1_ls.append(f1[0])
        
        dict_ls.append({
            "model_name": model_name,
            "data_name": data_name,
            "class": class_names[i],
            "Accuracy" : np.stack(acc_ls),
            "Sensitivty": np.stack(sen_ls),
            "Specificity": np.stack(spe_ls),
            "Precision": np.stack(pre_ls),
            "AUC": np.stack(auc_ls),
            "F1-Score": np.stack(f1_ls),
        })
    return dict_ls


# %%
dict_ls = []
model_name = ["VGG19", 'VGG16', 'DenseNet121']
data_name = ["non_tilt", 'tilt', 'All']
csvs = [csv_non_tilt, csv_tilt, csv]
for model in model_name:
    for i in range(3):
        dict_ls += cal_metrics_for_each_class(proba, model, data_name[i], csvs[i])
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
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr[i], tpr[i], _ = metrics.roc_curve(csv_tilt['class']==2, proba['tilt']['VGG19'][0, :, 2])
roc_auc[i] = metrics.auc(fpr[i], tpr[i])
# %%
import matplotlib.pyplot as plt
plt.figure()
lw = 2
plt.plot(
    fpr[2],
    tpr[2],
    color="darkorange",
    lw=lw,
    label="ROC curve (area = %0.2f)" % roc_auc[2],
)
plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating characteristic example")
plt.legend(loc="lower right")
plt.show()
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
