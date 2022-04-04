# %%
import pandas as pd
import numpy as np
from glob import glob
import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
# %%
csv = pd.read_csv('./data/data_2022_01_19.csv')
# %%
csv['img_shape'] = csv['filename'].map(lambda x: cv2.imread(x).shape)
# %%
sm = csv.query("sm == 'Y'")
opa = csv.query("sm == 'N'")
# %%
t = sm[sm['img_shape'] == (1536, 2048, 3)]
# %%
img = cv2.imread(t['filename'].iloc[0])
plt.imshow(img)
# %%
opa['img_shape'].value_counts()
# %%
# for shape in opa['img_shape'].value_counts().index:
#     t = opa[opa['img_shape'] == shape]
#     print(shape)
#     img = cv2.imread(t['filename'].iloc[0])
#     plt.figure()
#     plt.imshow(img)
# %%
t = opa[opa['img_shape'] == (1312, 1680, 3)]
img = cv2.imread(t['filename'].iloc[2])
plt.imshow(img)
# %%
img[1312-80:, :100] = 0
plt.imshow(img)
# %%
plt.imshow(cv2.resize(img, (224, 224)))
# %%
def opa_image_processing(image):
    img = image.copy()
    if img.shape == (1068, 1500, 3):
        img[:50, :200] = 0
        img[900:, :250] = 0
        img[900:, 1150:] = 0
        img = img[50:1018, 150:1350]
    elif img.shape == (1212, 1500, 3):
        img[:50, :170] = 0
        img[1150:, :170] = 0
        img[:50, 1350:] = 0
        img[1150:, 1350:] = 0
    else:
        h, _, _ = img.shape 
        img[h-80:, :100] = 0
    return img

def sm_image_processing(image, path):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img ,30,255,0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: x.shape, reverse=True)
    if contours[0].shape[0] >= 30:
        x,y,w,h = cv2.boundingRect(contours[0])
        img = image[y:y+h, x-30:x+w+30]
    else:
        print("Can't draw contours, %s"%path)
        return path
    return img
def process(path, sm):
    save_path = os.path.join(os.path.dirname(path), 
                             os.path.basename(path).split('.')[0]+'_cropped.jpg')
    image = cv2.imread(path)
    if sm == 'Y':
        val = sm_image_processing(image, path)
    else:
        val = opa_image_processing(image)
    if type(val) != str:
        val = cv2.resize(val, (224, 224))
        cv2.imwrite(save_path, val)
        return True
    else:
        return False
# %%
tqdm.pandas()
error_ls = csv.progress_apply(lambda x: process(x['filename'], x['sm']), axis=1)
# %%
csv['filename_crop'] = csv['filename'].map(lambda x: x[:-4]+'_cropped.jpg')
# %%
from shutil import copy
csv['filename_crop'].map(lambda x: copy(x, './images/%s'%os.path.basename(x)))

# %%
