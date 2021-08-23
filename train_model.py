from utils import *
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from official.nlp import optimization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import argparse, pickle

parser = argparse.ArgumentParser(description = "TrainArgs");
parser.add_argument('--type', choices=['all', 'tilt', 'non_tilt'], help='')
parser.add_argument('--gpu_id', type=int, help='')
args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    tf.config.experimental.set_visible_devices(gpus[args.gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu_id], True)

csv = pd.read_csv('./data.csv', index_col=0)
csv['class'] = csv['class'].astype(str)

csv, test = train_test_split(csv, test_size=0.2, 
                             random_state=1004, 
                             stratify=csv['class'])

csv_tilt = csv[csv['tilt'] == 1]
csv_not_tilt = csv[csv['tilt'] == 0]
model_name = ['vgg16', 'vgg', 'dense121']

if args.type == 'all':
    save_path = '/mnt/hdd/spow12/fundus/diagnosis/with_test/model_all'
    use_csv = csv
elif args.type == 'tilt':
    save_path = '/mnt/hdd/spow12/fundus/diagnosis/with_test/model_tilt'
    use_csv = csv_tilt
elif args.type == 'non_tilt':
    save_path = '/mnt/hdd/spow12/fundus/diagnosis/with_test/model_non_tilt'
    use_csv = csv_non_tilt
    

    
vgg16_histories = train_model(concat=True, normalize=True, 
                              save_path=save_path, data=use_csv,
                              model_name='vgg16')
vgg19_histories = train_model(concat=True, normalize=True, 
                              save_path=save_path, data=use_csv,
                              model_name='vgg')
dense121_histories = train_model(concat=True, normalize=True, 
                              save_path=save_path, data=use_csv,
                              model_name='dense121')

for histories in [vgg16_histories, vgg19_histories, dense121_histories]:
    for idx, history in enumerate(histories):
        with open(save_path+'/model_{0}_history_{1}.pckl'.format(mode_name[i], idx), 'wb') as fil:
            pickle.dump(history.history, fil)
