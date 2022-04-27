# %%
import pandas as pd
import datatable
# %%

csv = datatable.fread("/mnt/hdd/spow12/work/SMC_2019-11-155_과제_원내용_안저AI판독/DATA_OPH1415.csv").to_pandas()
# %%
data = pd.read_csv('./data/data_2022_04_11_final.csv')
# %%
sm = data.query('sm == "Y"')

# %%
