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
sm = csv.query("sm == 'Y'")
opa = csv.query("sm == 'N'")
# %%
img_rgb = cv2.imread(sm['filename'].iloc[1])
img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img ,30,255,0)
contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x: x.shape, reverse=True)
plt.imshow(img)
# %%

# (x,y),radius = cv2.minEnclosingCircle(contours[1])
# center = (int(x),int(y))
# radius = int(radius)
# plt.imshow(cv2.circle(img,center,radius,(255,255,255),5))

# %%
x,y,w,h = cv2.boundingRect(contours[0])
plt.imshow(img_rgb[y:y+h, x-30:x+w+30])
# %%
cv2.imwrite('./temp.jpg',img_rgb[y:y+h, x-30:x+w+30])
# %%

tqdm.pandas()
def img_cropping(path):
    try:
        img_rgb = cv2.imread(path)
        img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img ,30,255,0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: x.shape, reverse=True)
        if contours[0].shape[0] >= 30:
            x,y,w,h = cv2.boundingRect(contours[0])
            cv2.imwrite(os.path.join(os.path.dirname(path), os.path.basename(path).split('.')[0]+'_cropped.jpg')
                        ,img_rgb[y:y+h, x-30:x+w+30])
        else:
            print("This image is something wrong")
            print(path)
    except:
        print("This image is something wrong")
        print(path)

sm['filename'].map(lambda x: img_cropping(x))
        

# %%
with open('./error.txt', 'r') as f:
    txt = f.readlines()
# %%
img_rgb = cv2.imread(txt[3][:-1])
img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img ,30,255,0)
contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x: x.shape, reverse=True)
plt.imshow(img)
# %%

temp = (sm['filename'].map(lambda x: 'sm' in x)) | (sm['filename'].map(lambda x: '2020_12_21' in x))
# %%
wtf = sm[~temp]
# %%
wtf['filename'].iloc[1]
# %%
path = csv[csv['filename'].map(lambda x: '29333276_OD_P001.jpg' in x)]['filename'].iloc[0]
# path = csv[csv['filename'].map(lambda x: '26017799_OS_P002.jpg' in x)]['filename'].iloc[0]
img_rgb = cv2.imread(path)
img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img ,10,255,0)
contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x: x.shape, reverse=True)
print(img.shape)
plt.imshow(img)
# # %%
# plt.imshow(img[:50, :200])
# # %%
# plt.imshow(img[900:, :250])
# # %%
# plt.imshow(img[900:, 1150:])
# # %%
# img[:50, :200] = 0
# %%
path = csv[csv['filename'].map(lambda x: '26281554_OD_S001.jpg' in x)]['filename'].iloc[0]
img_rgb = cv2.imread(path)
img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img ,10,255,0)
contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x: x.shape, reverse=True)
print(img.shape)
plt.imshow(img)
# img[1200:, :100] = 0

# %%
path = csv[csv['filename'].map(lambda x: '12945626_OD_G001.jpg' in x)]['filename'].iloc[0]
img_rgb = cv2.imread(path)
img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(img ,10,255,0)
contours, hierachy = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=lambda x: x.shape, reverse=True)
print(img.shape)
plt.imshow(img)
# # %%
# plt.imshow(img[:50, :170])
# # %%
# plt.imshow(img[1150:, :170])
# # %%
# plt.imshow(img[:50, 1350:])
# # %%
# plt.imshow(img[1150:, 1350:])
# # %%

# %%
def image_processing(image):
    img = image.copy()
    if img.shape == (1068, 1500, 3):
        img[:50, :200] = 0
        img[900:, :250] = 0
        img[900:, 1150:] = 0
        img = img[50:1018, 150:1350]
    elif img.shape == (1282, 1700, 3):
        img[1200:, :100] = 0
    elif img.shape == (1212, 1500, 3):
        img[:50, :170] = 0
        img[1150:, :170] = 0
        img[:50, 1350:] = 0
        img[1150:, 1350:] = 0
    elif img.shape == (1125, 1500, 3):
        img = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(img ,30,255,0)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=lambda x: x.shape, reverse=True)
        if contours[0].shape[0] >= 30:
            x,y,w,h = cv2.boundingRect(contours[0])
            img = image[y:y+h, x-30:x+w+30]
        else:
            return path
    else:
        print("Wrong img shape")
        return path
    return img