import numpy as np
from utils import *
from sklearn.metrics import f1_score, accuracy_score, auc, classification_report, roc_curve
from sklearn.metrics import recall_score, precision_score, RocCurveDisplay, roc_auc_score

def print_evaluate_metrics(weight_pathes, data, dataset_name, model_names, out_dim=4, 
                           concat=True, training_data=None):
    dict_ls = []
    for i in range(3):
        model = make_model(model_names[i], concat=concat, out_dim=out_dim)
        weight_path = weight_pathes+model_names[i]
        precision, recall, f1, acc, auc = cal_indexes(model, weight_path, data)
        dict_ls.append({'precision': '%.4f' % np.mean(precision) + '±%.4f' % np.std(precision),
                        'accuracy': '%.4f' % np.mean(acc) + '±%.4f' % np.std(acc),
                        'recall': '%.4f' % np.mean(recall) + '±%.4f' % np.std(recall),
                        'f1_score':  '%.4f' % np.mean(f1) + '±%.4f' % np.std(f1),
                        'auc': '%.4f' % np.mean(auc) + '±%.4f' % np.std(auc),
                        'model_name': model_names[i], 'dataset': dataset_name,
                        'training_data': training_data})
    return dict_ls


def cal_indexes(model, weight_path, data):
    precision = []
    f1 = []
    recall = []
    acc = []
    auc = []
    val_datagen = FixedImageDataGenerator(
        #Your code here. Should at least have a rescale. Other parameters can help with overfitting
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
    for i in range(5):

        y_true = data.copy()
        y_true = y_true['class'].astype(float)
        model.load_weights(weight_path+'_{0}.h5'.format(i))
        prob_pred = model.predict(val_generator)
        pred = np.argmax(prob_pred, axis=1)
        precision.append(precision_score(
            y_true, y_pred=pred, average='weighted'))
        recall.append(recall_score(y_true, y_pred=pred, average='weighted'))
        f1.append(f1_score(y_true, y_pred=pred, average='weighted'))
        acc.append(accuracy_score(y_true, pred))
        auc.append(roc_auc_score(tf.one_hot(y_true, 4).numpy(), prob_pred, multi_class='ovr'))
    return precision, recall, f1, acc, auc
