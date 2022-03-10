# %%
from utils import *
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse, pickle

parser = argparse.ArgumentParser(description = "TrainArgs")
parser.add_argument('--model', choices=['vgg', 'vgg16', 'dense121'], help='')
parser.add_argument('--gpu_id', type=int, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--save_path', type=str, required=True)
args = parser.parse_args()


gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    tf.config.experimental.set_visible_devices(gpus[args.gpu_id], 'GPU')
    tf.config.experimental.set_memory_growth(gpus[args.gpu_id], True)
    
csv = pd.read_csv('./data/data_2022_01_19.csv')
csv, test = train_test_split(csv, test_size=0.2, 
                             random_state=1004, 
                             stratify=csv['class'])

csv['class'] = csv['tilt'].astype(str)
csv = csv.reset_index(drop=True)
# %%
histories = train_model(concat=True, normalize=True, 
                        save_path=args.save_path, data=csv,
                        model_name=args.model, out_dim=2, batch_size=args.batch_size)

for idx, history in enumerate(histories):
    with open(args.save_path+'/model_{0}_history_{1}.pckl'.format(args.model, idx), 'wb') as fil:
        pickle.dump(history.history, fil)
