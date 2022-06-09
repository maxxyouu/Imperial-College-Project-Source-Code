import json
import numpy as np
import os
import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
import cv2
################
# config
mng_store = './input/to_annotate/MNG/'
gbm_store = './input/to_annotate/GBM/'
data_json = './input/df.json'
anno_json = './input/anno_clean.json'
################

def load_img(path):
    """
    Load an img with opencv...
    """
    return(cv2.imread(path).copy())
 

def add_bbox(img, anno):
    """
    show bbox
    anno:
    Index(['id', 'iscrowd', 'image_id', 
    'category_id', 'segmentation', 'bbox',
    'area', 'category_name'],
    """
    for i, row in anno.iterrows():
        x, y, w, h = row.bbox
        x, y, w, h = int(x), int(y), int(w), int(h)
        cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255))
        cv2.putText(img, row.category_name, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36,255,12), 1)
    cv2.imshow('imgs', img)
    cv2.waitKey()


def show_one(row, anno):
    """
    draw bbox for one row
    """
    # where to find
    if row.label == 'meningioma':
        path_base = mng_store
    if row.label == 'glioblastoma':
        path_base = gbm_store
    # load img
    img = load_img(os.path.join(path_base, row.file_name))
    add_bbox(img, anno[anno['image_id']==row['id']])
    # add bounding boxes
    # add_bbox(img, row)

if __name__ == '__main__':
    df = pd.read_json(data_json)
    anno = pd.read_json(anno_json)
    # show one row
    # show_one(df.loc[0, :], anno)
    for j, row in df.iterrows():
        if j>10:
            break
        show_one(row, anno)