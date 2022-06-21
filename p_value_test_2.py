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
from utils import cal_metrics_for_each_class
# %%
csv = pd.read_csv('./data/data_2022_04_11_final.csv')
dev, csv = train_test_split(csv, test_size=0.2, 
                             random_state=1004, 
                             stratify=csv['class'])

csv_tilt = csv[csv['tilt'] == 1]
csv_non_tilt = csv[csv['tilt'] == 0]
# %%
with open('./data/test_proba_dual.pckl', 'rb') as f:
    proba = pickle.load(f)
# %%

class_names = [
    'Normal', 
    "Glaucoma",
    "Optic disc pallor", 
    "Optic disc swelling"
]
dict_ls = []
model_name = ["VGG19", 'VGG16', 'DenseNet121']
data_name = ["non_tilt", 'tilt', 'All']
csvs = [csv_non_tilt, csv_tilt, csv]
# %%
for model in model_name:
    for i in range(3):
        dict_ls += cal_metrics_for_each_class(proba, model, data_name[i], csvs[i])
# %%
df = pd.DataFrame(dict_ls)

# %%
df.query("model_name == 'VGG16' and data_name != 'All'")

# %%
i = 2

non_tilt = df.query("model_name == '%s' and data_name == '%s' and label == '%s'" %(
    model_name[i], 'non_tilt', class_names[i]))['Accuracy'].iloc[0]

tilt = df.query("model_name == '%s' and data_name == '%s' and label == '%s'" % (
    model_name[i], 'tilt', class_names[i]))['Accuracy'].iloc[0]

if shapiro(non_tilt)[1] >= 0.05 and shapiro(tilt)[1] >= 0.05 and levene(non_tilt, tilt)[1] >= 0.05:
    print(stats.ttest_ind(non_tilt, tilt))
else:
    print(stats.wilcoxon(non_tilt, tilt))
# %%
metric_names = df.columns[3:]
# %%
metric_names
# %%
dict_p = []
class_idx = 0
metric_idx = 0

non_tilt = df.query("model_name == '%s' and data_name == '%s' and label == '%s'" %(
        model_name[i], 'non_tilt', class_names[class_idx]))[metric_names[metric_idx]].iloc[0]

tilt = df.query("model_name == '%s' and data_name == '%s' and label == '%s'" % (
    model_name[i], 'tilt', class_names[class_idx]))[metric_names[metric_idx]].iloc[0]

np.enon_tilt
# %%
def get_p_values(df, model_name, class_name, metrics):
    
    def get_value(df, model_name, class_name, metric_name):
        non_tilt = df.query("model_name == '%s' and data_name == '%s' and label == '%s'" %(
            model_name, 'non_tilt', class_name))[metric_name].iloc[0]
        tilt = df.query("model_name == '%s' and data_name == '%s' and label == '%s'" %(
            model_name, 'tilt', class_name))[metric_name].iloc[0]
        return non_tilt, tilt

    
    p_dicts = []
    for metric in metrics:
        p_dict = {
            "model_name" : model_name,
            "class_name" : class_name,
        }
        non_tilt, tilt = get_value(df, model_name, class_name, metric)
        # p_val, method = cal_p_value(non_tilt, tilt)
        if levene(non_tilt, tilt)[1] >= 0.05:
            p_val = stats.ttest_ind(non_tilt, tilt, equal_var=True)[1]
        else:
            p_val = stats.ttest_ind(non_tilt, tilt, equal_var=False)[1]
        p_dict["metric"] = metric
        p_dict["non_tilt"] = '%.4f' % np.mean(non_tilt) + '±%.4f' % np.std(non_tilt)
        p_dict["tilt"] = '%.4f' % np.mean(tilt) + '±%.4f' % np.std(tilt)
        p_dict['p_value'] = p_val
        # p_dict['method'] = method
        p_dicts.append(p_dict)
    return p_dicts
    
# %%
metric_names = ['Accuracy', 'Sensitivty', 'Specificity']
p_dict = []
for model in model_name:
    for label in class_names:
        p_dict += get_p_values(df, model, label, metric_names)
        
# %%
pd.DataFrame(p_dict).to_excel('./data/p_val.xlsx')
# %%
pd.DataFrame(p_dict)
# %%
stats.mannwhitneyu(non_tilt, tilt)
# %%
import pandas as pd
# %%ㅂ
t = pd.read_excel('./data/p_val.xlsx')
# %%
t['diff'] = t.apply(lambda x: float(x['non_tilt'][:6]) - float(x['tilt'][:6]), axis=1)

# %%
t.query('class_name == "Normal" and metric == "Sensitivty"')['diff'].mean()
# %%
