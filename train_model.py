# %%
from utils import *
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse, pickle

# %%
parser = argparse.ArgumentParser(description = "TrainArgs")
parser.add_argument(
    '--type', choices=['multitask', 'all', 'tilt', 'non_tilt'], help='')
parser.add_argument('--model', choices=['vgg', 'vgg16', 'dense121'], help='')
parser.add_argument('--gpu_id', type=int, required=True)
parser.add_argument('--multi_task', type=bool, default=False)
parser.add_argument('--batch_size', type=int, default=32)
args = parser.parse_args()

gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    tf.config.experimental.set_visible_devices(gpus[args.gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu_id], True)

csv = pd.read_csv('./data/data_2022_04_11_final.csv')
csv['class'] = csv['class'].astype(str)
csv['filename'] = csv['filename'].map(lambda x: x[:-4]+'_cropped.jpg')

csv, test = train_test_split(csv, test_size=0.2, 
                             random_state=1004, 
                             stratify=csv['class'])

csv_tilt = csv[csv['tilt'] == 1]
csv_not_tilt = csv[csv['tilt'] == 0]
# %%
if args.type == 'all':
    save_path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_all/'
    use_csv = csv
elif args.type == 'tilt':
    save_path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_tilt/'
    use_csv = csv_tilt
elif args.type == 'non_tilt':
    save_path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/model_non_tilt/'
    use_csv = csv_not_tilt
elif args.type == 'multitask':
    save_path = '/mnt/hdd/spow12/work/fundus/fundus_paper/model_weights/disease_classification/multi_task_model/'
    use_csv = csv
else:
    raise 'NonValidDataError'
    

use_csv = use_csv.reset_index(drop=True)
histories = train_model(concat=True, normalize=True, 
                              save_path=save_path, data=use_csv,
                              model_name=args.model, out_dim=4, batch_size=args.batch_size,
                              multi_task=args.multi_task)

for idx, history in enumerate(histories):
    with open(save_path+'/model_{0}_history_{1}.pckl'.format(args.model, idx), 'wb') as fil:
        pickle.dump(history.history, fil)
