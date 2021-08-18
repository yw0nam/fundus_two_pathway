from utils import *
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow.keras as keras
from official.nlp import optimization
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

csv = pd.read_csv('./data.csv', index_col=0)
csv['class'] = csv['class'].astype(str)
csv_tilt = csv[csv['tilt'] == 1]
csv_not_tilt = csv[csv['tilt'] == 0]

vgg16_histories = train_model(concat=True, normalize=True, 
                              save_path='./model_tilt', data=csv_tilt,
                              model_name='vgg16')
vgg19_histories = train_model(concat=True, normalize=True, 
                              save_path='./model_tilt', data=csv_tilt,
                              model_name='vgg')
dense121_histories = train_model(concat=True, normalize=True, 
                              save_path='./model_tilt', data=csv_tilt,
                              model_name='dense121')

vgg16_histories_non = train_model(concat=True, normalize=True, 
                              save_path='./model_non_tilt', data=csv_not_tilt,
                              model_name='vgg16')
vgg19_histories_non = train_model(concat=True, normalize=True, 
                              save_path='./model_non_tilt', data=csv_not_tilt,
                              model_name='vgg')
dense121_histories_non = train_model(concat=True, normalize=True, 
                              save_path='./model_non_tilt', data=csv_not_tilt,
                              model_name='dense121')