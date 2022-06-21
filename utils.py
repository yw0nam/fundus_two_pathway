from tensorflow.keras.models import Model
from official.nlp import optimization
from tensorflow.keras.layers import BatchNormalization, Dense, Input, Conv2D, Activation, Dropout, MaxPooling2D, GlobalAveragePooling2D, Flatten, Concatenate
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import metrics
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from tqdm import tqdm
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import os 
import seaborn as sns

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
            "label": class_names[i],
            "Accuracy" : np.stack(acc_ls),
            "Sensitivty": np.stack(sen_ls),
            "Specificity": np.stack(spe_ls),
            "Precision": np.stack(pre_ls),
            "AUC": np.stack(auc_ls),
            "F1-Score": np.stack(f1_ls),
        })
    return dict_ls

def dense_block(units, dropout=0.2, activation='relu', name='fc1'):

    def layer_wrapper(inp):
        x = Dense(units, name=name)(inp)
        x = BatchNormalization(renorm=True, name='{}_bn'.format(name))(x)
        x = Activation(activation, name='{}_act'.format(name))(x)
        x = Dropout(dropout, name='{}_dropout'.format(name))(x)
        return x

    return layer_wrapper

def make_model_pretrain(shape=(224, 224, 3), 
                        dropout=0.3, activation='relu', 
                        model_name='vgg',
                       concat=True,
                       out_dim=1, multi_task=False):
    
    inp = tf.keras.layers.Input(shape, dtype=tf.float32)
    if model_name == 'vgg':
        dense = tf.keras.applications.VGG19(include_top=False, input_shape=shape)
        for layer in dense.layers:
            layer.trainable = False
        dense_trainable = tf.keras.applications.VGG19(include_top=False, input_shape=shape)
        dense_trainable._name = 'trainable_VGG19'
    elif model_name == 'vgg16':
        dense = tf.keras.applications.VGG16(include_top=False, input_shape=shape)
        for layer in dense.layers:
            layer.trainable = False
        dense_trainable = tf.keras.applications.VGG16(include_top=False, input_shape=shape)
        dense_trainable._name = 'trainable_VGG16'
    elif model_name == 'dense121':
        dense = tf.keras.applications.DenseNet121(include_top=False, input_shape=shape)
        for layer in dense.layers:
            layer.trainable = False
        dense_trainable = tf.keras.applications.DenseNet121(include_top=False, input_shape=shape)
        dense_trainable._name = 'trainable_DenseNet121'
    elif model_name == 'dense169':
        dense = tf.keras.applications.DenseNet169(include_top=False, input_shape=shape)
        for layer in dense.layers:
            layer.trainable = False
        dense_trainable = tf.keras.applications.DenseNet169(include_top=False, input_shape=shape)
        dense_trainable._name = 'trainable_DenseNet169'
    elif model_name == 'dense201':
        dense = tf.keras.applications.DenseNet201(include_top=False, input_shape=shape)
        for layer in dense.layers:
            layer.trainable = False
        dense_trainable = tf.keras.applications.DenseNet201(include_top=False, input_shape=shape)
        dense_trainable._name = 'trainable'
    else:
        assert print('Wrong model name!')
    # x = tf.cast(inp, tf.float32)
    # x = tf.keras.applications.densenet.preprocess_input(inp)

    x_trainable = dense_trainable(inp)
    x = GlobalAveragePooling2D()(x_trainable)
    
    if concat:
        x_non = dense(inp)
        x_non = GlobalAveragePooling2D()(x_non)
        x = Concatenate()([x_non, x])
        
    x = dense_block(1024, dropout=dropout, activation=activation, name='fc1')(x)
    x = dense_block(512, dropout=dropout, activation=activation, name='fc2')(x)
    x = dense_block(64, dropout=dropout, activation=activation, name='fc3')(x)
    #     x = dense_block(64, dropout=0.25, activation='relu', name='fc3')(x)
    if multi_task:
        x_tilt = Dense(2, activation='softmax', name='y_tilt')(x)
        x_disease = Dense(out_dim, activation='softmax', name='y_disease')(x)
        x = [x_tilt, x_disease]
    else:
        if out_dim == 1:
            x = Dense(out_dim, activation='sigmoid')(x)
        else:
            x = Dense(out_dim, activation='softmax')(x)

    model = Model(inp, x)
    return model

def make_model(model_name, concat=False, 
               len_data=814, out_dim=1,
               multi_task=False):
    epochs = 100
    batch_size = 16
    image_size=(256, 256)
    init_lr = 1e-4
    model = make_model_pretrain(model_name=model_name,
                                concat=concat, out_dim=out_dim,
                                multi_task=multi_task)
    
    steps_per_epoch = len_data // batch_size
    num_train_steps = steps_per_epoch * epochs
    num_warmup_steps = int(0.1*num_train_steps)

    optimizer = optimization.create_optimizer(init_lr=init_lr,
                                              num_train_steps=num_train_steps,
                                              num_warmup_steps=num_warmup_steps,
                                              optimizer_type='adamw')

    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model

class FixedImageDataGenerator(ImageDataGenerator):
    def standardize(self, x):
        if self.featurewise_center:
            x = ((x/255.) - 0.5) * 2.
        return x
    
def train_model(concat, normalize, model_name, save_path, data, init_lr=1e-4, batch_size=32,
                epochs=100, out_dim=4, n_fold=5, class_weight=None, multi_task=False):
    histories = []
    image_size=(224, 224)
    skf = StratifiedKFold(n_splits=n_fold,
                          random_state=1004, shuffle=True)
    
    for cv_idx, data_index in enumerate(skf.split(data['filename'], data['class'])):
        train_index, val_index = data_index[0], data_index[1]
        train_image, val_image = data['filename'][train_index], data['filename'][val_index]
        train_label, val_label = data['class'][train_index], data['class'][val_index]
        train_tilt_label, val_tilt_label = data['tilt'][train_index], data['tilt'][val_index]
        
        steps_per_epoch = len(train_image) // batch_size
        num_train_steps = steps_per_epoch * epochs
        num_warmup_steps = int(0.1*num_train_steps)
        
        train = pd.DataFrame({"filename": train_image,
                            "class": train_label,
                            "tilt": train_tilt_label
                            })
        train['class'] = train['class'].astype(str)
        train['tilt'] = train['tilt'].astype(str)
        
        val = pd.DataFrame({"filename": val_image,
                            "class": val_label,
                            "tilt": val_tilt_label})
        val['tilt'] = val['tilt'].astype(str)
        
        train_datagen = FixedImageDataGenerator(
        #Your code here. Should at least have a rescale. Other parameters can help with overfitting
            featurewise_center=True if normalize else False,
            featurewise_std_normalization=True if normalize else False,
            brightness_range=[0.8, 1.2],
            width_shift_range=0.05,
            horizontal_flip=True,
            rescale=1./255
        )
        train_generator = train_datagen.flow_from_dataframe(
            train,
            target_size=image_size,
            batch_size=batch_size,
            y_col=['tilt', 'class'] if multi_task else 'class',
            class_mode='multi_output' if multi_task else 'categorical'
        )


        val_datagen = FixedImageDataGenerator(
        #Your code here. Should at least have a rescale. Other parameters can help with overfitting
            featurewise_center=True,
            featurewise_std_normalization=True,
            rescale=1./255
        )

        val_generator = val_datagen.flow_from_dataframe(
            val,
            target_size=image_size,
            batch_size=batch_size,
            y_col=['tilt', 'class'] if multi_task else 'class',
            class_mode='multi_output' if multi_task else 'categorical'
        )
        model = make_model_pretrain(dropout=0.3, activation='relu', 
                                    model_name=model_name, concat=concat,
                                   out_dim=out_dim, multi_task=multi_task)
        optimizer = optimization.create_optimizer(init_lr=init_lr,
                                                  num_train_steps=num_train_steps,
                                                  num_warmup_steps=num_warmup_steps,
                                                  optimizer_type='adamw')
        
        if multi_task:
            model.compile(optimizer=optimizer,
                loss={'y_tilt' :'categorical_crossentropy',
                    'y_disease': 'categorical_crossentropy'},
                loss_weights={'y_tilt': 0.5, 'y_disease': 1},
                metrics=['accuracy'])
            checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path+'/model_{0}_{1}.h5'.format(model_name, cv_idx),
                                                            verbose=2, monitor='val_y_disease_loss', save_best_only=True, mode='auto')
        else:
            model.compile(optimizer=optimizer,
                        loss='categorical_crossentropy',
                        metrics=['accuracy'])
            checkpoint = tf.keras.callbacks.ModelCheckpoint(save_path+'/model_{0}_{1}.h5'.format(model_name, cv_idx),
                                                            verbose=2, monitor='val_loss', save_best_only=True, mode='auto')

        if multi_task:
            history = model.fit(
                get_data_from_generator(train_generator),
                validation_data=get_data_from_generator(val_generator),
                steps_per_epoch=len(train_generator),
                validation_steps=len(val_generator),
                epochs=epochs, 
                callbacks=[checkpoint],
            )
        else:
            history = model.fit(
                train_generator,
                validation_data=val_generator,
                epochs=epochs,
                callbacks=[checkpoint],
                class_weight=class_weight
            )
        histories.append(history)
    return histories


def get_data_from_generator(generator, out_dim=[2, 4]):
    while(True):
        data = next(generator)
        yield data[0], [tf.one_hot(data[1][0], depth=out_dim[0]), tf.one_hot(data[1][1], depth=out_dim[1])]


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
            target_size=(224, 224),
            batch_size=32,
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
            viz = metrics.RocCurveDisplay.from_predictions(
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
        mean_auc = metrics.auc(mean_fpr, mean_tpr)
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
