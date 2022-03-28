from utils import *
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse, pickle
import numpy as np

def cal_pred(weight_path, data_generator, model_name, out_dict):
    preds = []
    model = make_model(model_name, concat=True, out_dim=2)
    for i in range(5):
        model.load_weights(weight_path+'model_%s_%d.h5'%(model_name, i))
        preds.append(model.predict(data_generator))
    out_dict[model_name] = np.stack(preds)
    return out_dict


def main():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        
    csv = pd.read_csv('./data/data_2022_01_19.csv')
    
    _, test = train_test_split(csv, test_size=0.2, 
                            random_state=1004, 
                            stratify=csv['class'])
    test['class'] = test['tilt'].astype(str)
    test = test.reset_index(drop=True)
    
    
    datagen = FixedImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        rescale=1./255
    )

    generator = datagen.flow_from_dataframe(
        test,
        target_size=(256, 256),
        batch_size=32,
        shuffle=False,
    )
    root_path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/tilt_classification/'
    pred_dicts = {}
    for model_name in ['vgg', 'vgg16', 'dense121']:
        pred_dicts = cal_pred(root_path, generator, model_name, pred_dicts)
    with open('./data/tilt_prediction.pckl', 'wb') as fil:
        pickle.dump(pred_dicts, fil)
        
if __name__ == '__main__':
    main()