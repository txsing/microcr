from dataset import generator as data_generation
from dataset import dataset

import pandas as pd
import shutil
import os

# If the dataset is already present setting this
# variable to True will force a total removal and recreation
remake_dataset = False
number_of_samples = 50000
dataset_config = data_generation.base_config


src_dir = './tmp/MICRST_BW'
re_org_dir = "./tmp/MICRST_BWTF"

if os.path.exists(src_dir) and os.path.isdir(src_dir):
    shutil.rmtree(src_dir)
if os.path.exists(re_org_dir) and os.path.isdir(re_org_dir):
    shutil.rmtree(re_org_dir)

print('Generating synthetic data in directory:', src_dir, '\nPlease wait')
data_generation.make_dataset(dataset_config, number_of_samples, force=remake_dataset, root_dir = src_dir)

# Re-organize the dataset into Tensorflow format
df = pd.read_csv(f"{folder}/labels.csv")
for idx in range(len(df)):
    f = df.iloc[idx, 0]
    label = df.iloc[idx, 1]
    tgt_path = copy_dir+'/'+label
    if not os.path.exists(tgt_path):
        os.makedirs(tgt_path)
    shutil.copy(f, tgt_path)
print("Re-organize Done")

