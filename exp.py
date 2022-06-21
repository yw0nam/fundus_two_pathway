# %%
import pandas as pd
# %%
csv = pd.read_csv('./data/test_metric_dual.csv')
# %%
csv['accuracy'] = csv['accuracy'].map(lambda x: "%.3f±%.3f"%(float(x.split('±')[0]), float(x.split('±')[1])))
csv['precision'] = csv['precision'].map(lambda x: "%.3f±%.3f"%(float(x.split('±')[0]), float(x.split('±')[1])))
csv['f1_score'] = csv['f1_score'].map(lambda x: "%.3f±%.3f" %
                    (float(x.split('±')[0]), float(x.split('±')[1])))
csv['auc'] = csv['auc'].map(lambda x: "%.3f±%.3f" %
                                (float(x.split('±')[0]), float(x.split('±')[1])))
# %%
csv.to_csv('./data/test_metric_dual_3.csv', index=False, encoding='cp949')

# %%
csv = pd.read_excel('./data/p_val.xlsx')
# %%
csv['non_tilt'] = csv['non_tilt'].map(lambda x: "%.3f±%.3f" % (
    float(x.split('±')[0]), float(x.split('±')[1])))
csv['tilt'] = csv['tilt'].map(lambda x: "%.3f±%.3f" % (
    float(x.split('±')[0]), float(x.split('±')[1])))
# %%
csv.to_excel('./data/p_val_3.xlsx')
# %%
