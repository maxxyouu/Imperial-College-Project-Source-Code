import json
import numpy as np
import os
import sys
import glob
import pandas as pd
from requests import head
################
data_csv = './input/ex-vivo-split/split.csv'
################
# open annotations and store in data variable
fp = '/media/alfie/Storage/Clinical_Data/pCLE/cleopatra/ex-vivo/GBM_meningioma_annotations/labels_annotations-all-final_2022-02-23-09-43-09.json'
with open(fp) as f:
    data = json.load(f)
    f.close()
print(f"Data available: {data.keys()}")
print(f"Categories in file: {data['categories']}")
print('################')
categories = {}
# assign categories
for cat in data['categories']:
    categories[cat['id']] = cat['name']
################
# data csv
df = pd.read_csv(data_csv)
if 'anno_name' in df.columns:
    raise Exception('Annotations already been loaded...')
df['anno_name'] = df['name_video'].apply(lambda x: x.replace(' ', '').strip())+'_'+df['n_slice'].astype(str)+'.png'
################
# load coco data frame
coco = pd.DataFrame(data['images'])
annotations = pd.DataFrame(data['annotations'])
annotations['category_name'] = annotations['category_id'].apply(lambda x: categories[x])
################
coco_final = pd.merge(df, coco, on='file_name')
coco_final.to_csv(data_csv, header=False)
annotations.to_json('./input/anno.json')
clean_anno = annotations[annotations.category_name != '??'].reset_index(drop=True)
clean_anno.to_json('./input/anno_clean.json')
print('Loaded file to df.json...')
