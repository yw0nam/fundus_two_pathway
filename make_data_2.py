# %%
import pandas as pd
from glob import glob
import numpy as np
import os
# %%
root_path =  '/mnt/hdd/spow12/work/fundus/2022_01_18/Ophthalmology/'
sm_root_path =  '/mnt/hdd/spow12/work/fundus/2022_01_18/sm/'
# %%
dict_ls = []
for disease in ['G', 'P', 'S']:
    non_tilt = glob(os.path.join(root_path, disease) +'/nontilt_*/*')
    tilt = glob(os.path.join(root_path, disease)+'/tilt_*/*')
    for path in non_tilt:
        dict_ls.append({
            'class' : disease,
            'tilt' : 0,
            'filename': path,
            'sm' : 'N'
        })
    for path in tilt:
        dict_ls.append({
            'class' : disease,
            'tilt' : 1,
            'filename': path,
            'sm' : 'N'
        })
# %%
def sm_make_dict(pathes, tilt, disease):
    dic_ls = []
    for path in pathes:
        dic_ls.append({
            'class' : disease,
            'tilt' : tilt,
            'filename': path,
            'sm' : 'Y'     
        })
    return dic_ls
# %%
p_non_tilt = glob('/mnt/hdd/spow12/work/fundus/2022_01_18/sm/P_nontilt/*')
normal_tilt = glob('/mnt/hdd/spow12/work/fundus/2022_01_18/sm/normal_tilted/*')
p_tilt = glob('/mnt/hdd/spow12/work/fundus/2022_01_18/sm/P_tilted/*')
g_tilt = glob('/mnt/hdd/spow12/work/fundus/2022_01_18/sm/G_tilt/*')
s_non_tilt = glob('/mnt/hdd/spow12/work/fundus/2022_01_18/sm/S_nontilt/*')
normal_nontilt = glob('/mnt/hdd/spow12/work/fundus/2022_01_18/sm/normal_nontilt/*')
# %%
dict_ls += sm_make_dict(p_non_tilt, 0, 'P')
dict_ls += sm_make_dict(p_tilt, 1, 'P')
dict_ls += sm_make_dict(normal_tilt, 1, 'N')
dict_ls += sm_make_dict(normal_nontilt, 0, 'N')
dict_ls += sm_make_dict(g_tilt, 1, 'G')
dict_ls += sm_make_dict(s_non_tilt, 0, 'S')
# %%
non_tilt_normal = glob('/mnt/hdd/spow12/work/fundus/images/2020_12_21/non */*')
tilt_normal = glob('/mnt/hdd/spow12/work/fundus/images/2020_12_21/tilted/*')
# %%
dict_ls += sm_make_dict(non_tilt_normal, 0, 'N')
# OH MY GOD
# dict_ls += sm_make_dict(non_tilt, 1, 'N')
dict_ls += sm_make_dict(tilt_normal, 1, 'N')
# %%
df = pd.DataFrame(dict_ls)
# %%
with open('./error.txt', 'r') as f:
    errors = f.readlines()
# %%
indexs = []
for error in errors:
    indexs += df[df['filename'].map(lambda x: error[:-1] in x)].index.to_list()
df = df.query("index not in @indexs")
# %%
df['class'] = df['class'].replace({'N': 0, 'G' : 1, 'P' : 2, 'S':3})
df.to_csv('./data/data_2022_01_19.csv', index=False)
# %%
import shutil
df['filename'].map(lambda x: shutil.copy(x, './images/%s'%os.path.basename(x)))
# %%
