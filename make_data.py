import tensorflow as tf
import numpy as np
import pandas as pd
from glob import glob
import tensorflow.keras as keras
from utils import *
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
  # 텐서플로가 첫 번째 GPU만 사용하도록 제한
    tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[1], True)
    
df_S = pd.DataFrame(columns=['filename', 'class', 'tilt'])
df_P = pd.DataFrame(columns=['filename', 'class', 'tilt'])
df_G = pd.DataFrame(columns=['filename', 'class', 'tilt'])
df_H = pd.DataFrame(columns=['filename', 'class', 'tilt'])

csv = pd.read_excel('/mnt/hdd/spow12/fundus/diagnosis_2021_08_04/녹내장 AI 명단 1차검수.xlsx', index_col=0)
csv_G = csv.query('시신경분류_1녹내장_2부종_3창백 == 1 and 사용여부 == 1')

root_path = '/mnt/hdd/spow12/fundus/diagnosis_2021_08_04/G/'

result = []
for id in csv_G['ID']:
    result += glob(root_path + str(id) +'*.jpg')

df_G['filename'] = result
df_G['class'] = 1
df_H['filename'] = glob('/mnt/hdd/spow12/fundus/diagnosis_2021_08_04/H/*.jpg')
df_H['class'] = 4
df_S['filename'] = glob('/mnt/hdd/spow12/fundus/diagnosis_2021_08_04/S/*.jpg')
df_S['class'] = 3
df_P['filename'] = glob('/mnt/hdd/spow12/fundus/diagnosis_2021_08_04/P/*.jpg')
df_P['class'] = 2

df = pd.concat([df_G, df_H, df_P, df_S])
df = df.reset_index(drop='index')
df['class'] = df['class'].astype(str)

val_datagen = FixedImageDataGenerator(
#Your code here. Should at least have a rescale. Other parameters can help with overfitting
    featurewise_center=True,
    featurewise_std_normalization=True,
    rescale=1./255
)

val_generator = val_datagen.flow_from_dataframe(
   df,
    target_size=(256, 256),
    batch_size=1,
    shuffle=False
)

model = make_model('vgg16', concat=True)
model.load_weights('/mnt/hdd/spow12/fundus/model_dual_normalize/model_vgg16_2.h5')
prediction = model.predict(val_generator)
temp = prediction.copy()
temp = np.where(temp < 0.5, 0, 1)
df['tilt'] = temp

non_tilt_normal = glob('/mnt/hdd/spow12/fundus/images/2020_12_21/non */*')
tilt_normal = glob('/mnt/hdd/spow12/fundus/images/2020_12_21/tilted/*')
df_normal_tilt = pd.DataFrame(columns=['filename', 'class', 'tilt'])
df_normal_non_tilt = pd.DataFrame(columns=['filename', 'class', 'tilt'])

df_normal_tilt['filename'] = tilt_normal
df_normal_tilt['class'] = 0
df_normal_tilt['tilt'] = 1

df_normal_non_tilt['filename'] = non_tilt_normal
df_normal_non_tilt['class'] = 0
df_normal_non_tilt['tilt'] = 0

df = pd.concat([df, df_normal_non_tilt, df_normal_tilt])
df = df.reset_index(drop='index')
df['class'] = df['class'].astype(str)
df.to_csv('./data.csv')