# %%
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from tqdm import tqdm
import numpy as np
import os
from sklearn.model_selection import train_test_split

# %%
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
# 텐서플로가 첫 번째 GPU만 사용하도록 제한
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[1], True)
    
csv = pd.read_csv('./data/data_2022_04_11_final.csv')
csv['class'] = csv['class'].astype(str)
csv['filename'] = csv['filename'].map(lambda x: x[:-4]+'_cropped.jpg')

_, test = train_test_split(csv, test_size=0.2,
                            random_state=1004,
                            stratify=csv['class'])

csv_tilt = test[test['tilt'] == 1]
csv_not_tilt = test[test['tilt'] == 0]

# %%
curve_maker = cv_roc_curve()
pred_dicts = {}

# Front: Modeling, Behind: Test
# %%
path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_non_tilt/model_'
curve_maker.set_params_and_predict(data=csv_tilt,
                                train_data_name="Non tilted disc",
                                test_data_name="Tilted disc",
                                root_path=path)

curve_maker.draw_full_graph('./figures/Paper/roc/')
pred_dicts['NT_T'] = curve_maker.pred_dicts

curve_maker.set_params_and_predict(data=csv_not_tilt,
                                train_data_name="Non tilted disc",
                                test_data_name="Non tilted disc",
                                root_path=path)

curve_maker.draw_full_graph('./figures/Paper/roc/')
pred_dicts['NT_NT'] = curve_maker.pred_dicts
# %%
path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_tilt/model_'
curve_maker.set_params_and_predict(data=csv_not_tilt,
                                train_data_name="Tilted disc",
                                test_data_name="Non tilted disc",
                                root_path=path)

curve_maker.draw_full_graph('./figures/Paper/roc/')
pred_dicts['T_NT'] = curve_maker.pred_dicts

curve_maker.set_params_and_predict(data=csv_tilt,
                                train_data_name="Tilted disc",
                                test_data_name="Tilted disc",
                                root_path=path)
curve_maker.draw_full_graph('./figures/Paper/roc/')
pred_dicts['T_T'] = curve_maker.pred_dicts
# %%
path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_all/model_'
curve_maker.set_params_and_predict(data=test,
                                train_data_name="All",
                                test_data_name="All",
                                root_path=path)
curve_maker.draw_full_graph('./figures/Paper/roc/')
pred_dicts['A_A'] = curve_maker.pred_dicts

curve_maker.set_params_and_predict(data=csv_tilt,
                                train_data_name="All",
                                test_data_name="Tilted disc",
                                root_path=path)
curve_maker.draw_full_graph('./figures/Paper/roc/')
pred_dicts['A_T'] = curve_maker.pred_dicts

curve_maker.set_params_and_predict(data=csv_not_tilt,
                                train_data_name="All",
                                test_data_name="Non tilted disc",
                                root_path=path)
curve_maker.draw_full_graph('./figures/Paper/roc/')
pred_dicts['A_NT'] = curve_maker.pred_dicts

# %%
curve_maker.draw_full_graph('./figures/Paper/roc/')

# %%
pred_dicts.keys()
# %%
import pickle
# with open('./data/test_proba_dual_2.pckl', 'wb') as fil:
#         pickle.dump(pred_dicts, fil)
# %%
with open('./data/test_proba_dual_2.pckl', 'rb') as f:
    pred_dicts = pickle.load(f)

# %%
def draw_roc_curve_respect_to_data(pred_dicts: dict, 
                                label:int, 
                                label_name: str,
                                save_path: str,
                                colors=['b', 'g', 'r']
                                ):
    def get_cv_roc(y_true, y_pred):
        _, ax = plt.subplots()
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for i in range(5):
            viz = metrics.RocCurveDisplay.from_predictions(
                y_true, y_pred[i, :], ax=ax, name="ROC fold %s" % i)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        return tprs, aucs
    
    def draw_roc_graph(tprs, aucs, ax, name, color='b', linestyle=None):
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            label=r"%s (AUC = %0.3f $\pm$ %0.3f)" % (name, mean_auc, std_auc),
            linestyle=linestyle,
            lw=2,
            alpha=0.8,
        )

    fig, ax = plt.subplots(figsize=(12, 10))
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)
    plt.xlabel("False Positive Rate", fontsize=25)
    plt.ylabel("True Positive Rate", fontsize=25)
    
    for i, dev_name in enumerate(['A_NT','A_T', 'NT_NT', 'NT_T','T_NT', 'T_T']):
        y_pred = pred_dicts[dev_name]['DenseNet121']
        y_true = csv_not_tilt if i % 2 == 0 else csv_tilt
        linestyle = None if i % 2 == 0 else '--'
        y_true = tf.one_hot(y_true['class'].astype(float), 4).numpy()
        
        if i <= 1:
            color = colors[0]
        elif 1 < i  and i <=3:
            color = colors[1]
        else:
            color = colors[2]
        tprs, aucs = get_cv_roc(y_true[:, label], y_pred[:, :, label])
        draw_roc_graph(tprs, aucs, ax, dev_name, color=color, linestyle=linestyle)

    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="k", alpha=0.8)
    ax.set(
        xlim=[-0.05, 1.05],
        ylim=[-0.05, 1.05],
    )
    ax.legend(loc="lower right", prop={'size':25})
    fig.savefig(os.path.join(save_path, "%s_%s.png" %
                        ('DenseNet121', label_name)))
    plt.show()
# %%
draw_roc_curve_respect_to_data(pred_dicts, 0, 'Normal', save_path='./figures/Paper/roc_2/')
# %%
draw_roc_curve_respect_to_data(pred_dicts, 1, 'Glaucoma', save_path='./figures/Paper/roc_2/')
draw_roc_curve_respect_to_data(pred_dicts, 2, 'Pale', save_path='./figures/Paper/roc_2/')
draw_roc_curve_respect_to_data(pred_dicts, 3, 'Swelling', save_path='./figures/Paper/roc_2/')
# %%
