# %%
import matplotlib.pyplot as plt
import pandas as pd
from utils import *
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, auc, classification_report, roc_curve
from sklearn.metrics import recall_score, precision_score, RocCurveDisplay
import os
from sklearn.model_selection import train_test_split
# %%
class cv_roc_curve:
    def __init__(self,
                 labels=['N', 'G', 'P', 'S'],
                 model_names=['vgg', 'vgg16', 'dense121'],
                 plot_model_names=['VGG19', 'VGG16', 'DenseNet121'],
                 colors=['b', 'g', 'r', 'y']):
        self.labels = labels
        self.model_names = model_names
        self.plot_model_names = plot_model_names
        self.colors = colors

    def set_params_and_predict(self, data, dataset_name, root_path):
        self.dataset_name = dataset_name
        self.root_path = root_path

        val_datagen = FixedImageDataGenerator(
            featurewise_center=True,
            featurewise_std_normalization=True,
            rescale=1./255
        )

        val_generator = val_datagen.flow_from_dataframe(
            data,
            target_size=(256, 256),
            batch_size=1,
            shuffle=False
        )
        self.pred_dicts = {}
        for i in tqdm(range(3)):
            model = make_model(self.model_names[i], concat=True, out_dim=4)
            weight_path = self.root_path + self.model_names[i]
            preds = []
            for j in range(5):
                model.load_weights(weight_path+'_{0}.h5'.format(j))
                prediction = model.predict(val_generator)
                preds.append(prediction)
            preds = np.stack(preds)
            self.pred_dicts[self.plot_model_names[i]] = preds

        y_true = data.copy()
        y_true = y_true['class'].astype(float)
        self.y_true = tf.one_hot(y_true, 4).numpy()

    @staticmethod
    def get_cv_roc(y_true, y_pred):
        _, ax = plt.subplots()
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)
        for i in range(5):
            viz = RocCurveDisplay.from_predictions(
                y_true, y_pred[i, :], ax=ax, name="ROC fold %s" % i)
            interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
            interp_tpr[0] = 0.0
            tprs.append(interp_tpr)
            aucs.append(viz.roc_auc)
        return tprs, aucs

    @staticmethod
    def draw_roc_graph(tprs, aucs, ax, name, color='b', std_label=False):
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        ax.plot(
            mean_fpr,
            mean_tpr,
            color=color,
            label=r"%s (AUC = %0.3f $\pm$ %0.3f)" % (name, mean_auc, std_auc),
            lw=2,
            alpha=0.8,
        )

#         ax.fill_between(
#             mean_fpr,
#             tprs_lower,
#             tprs_upper,
#             color="grey",
#             alpha=0.2,
#             label=r"$\pm$ 1 std. dev." if std_label else None,
#         )
    def draw_full_graph(self, save_path):
        for model_name in self.plot_model_names:
            fig, ax = plt.subplots(figsize=(12, 12))
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.xlabel("False Positive Rate", fontsize=25)
            plt.ylabel("True Positive Rate", fontsize=25)
            for i in range(4):
                tprs, aucs = cv_roc_curve.get_cv_roc(
                    self.y_true[:, i], self.pred_dicts[model_name][:, :, i])
#                 draw_roc_graph(tprs, aucs, ax, labels[i], color=colors[i], std_label= True if i == 3 else False)
                cv_roc_curve.draw_roc_graph(
                    tprs, aucs, ax, self.labels[i], color=self.colors[i])

            ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="k", alpha=0.8)
            ax.set(
                xlim=[-0.05, 1.05],
                ylim=[-0.05, 1.05],
                title="Dataset: %s  Model: %s" % (
                    self.dataset_name, model_name),
            )
            ax.legend(loc="lower right", prop={'size': 20})
            # plt.savefig('./figures/AI_all.png', facecolor='#eeeeee')
            fig.savefig(os.path.join(save_path, "%s_%s.png" %
                        (self.dataset_name, model_name)))
            plt.show()


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

curve_maker = cv_roc_curve()
pred_dicts = {}

path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_non_tilt/model_'
curve_maker.set_params_and_predict(data=csv_not_tilt,
                                   dataset_name="Non-Tilt",
                                   root_path=path)

pred_dicts['non_tilt'] = curve_maker.pred_dicts
curve_maker.draw_full_graph('./figures/Paper/roc/')

# %%
path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_tilt/model_'
curve_maker.set_params_and_predict(data=csv_tilt,
                                   dataset_name="Tilt",
                                   root_path=path)

pred_dicts['tilt'] = curve_maker.pred_dicts
curve_maker.draw_full_graph('./figures/Paper/roc/')
# %%
path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_all/model_'
curve_maker.set_params_and_predict(data=test,
                                   dataset_name="All",
                                   root_path=path)

pred_dicts['All'] = curve_maker.pred_dicts

# %%
curve_maker.draw_full_graph('./figures/Paper/roc/')

# %%
pred_dicts.keys()
# %%
import pickle
with open('./data/test_proba_dual.pckl', 'wb') as fil:
        pickle.dump(pred_dicts, fil)
# %%
with open('./data/test_proba_dual.pckl', 'rb') as f:
    temp = pickle.load(f)
# %%
temp
# %%
