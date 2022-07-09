import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import json
import cv2
import numpy as np
import shutil
from PIL import Image

def scale(img):
    """
    min max scale img
    """
    max_ = img.max()
    min_ = img.min()
    return((img-min_)/((max_-min_)))


def make_mask(img, anno):
    """
    make mask from annoatations
    """
    # segmentation
    assert(len(anno['segmentation']) == 1)
    seg = list(map(int, anno['segmentation'][0])) # convert the floating point to integer and back to list

    # x's from list
    x, y = [], []
    # append x's and y's
    for idx, p in enumerate(seg):
        if idx%2==0:
            x.append(p)
        else:
            y.append(p)
    # points for poly
    points = np.array(list(zip(x,y)))
    # fill convex hull
    mask = np.zeros_like(img)
    cv2.fillConvexPoly(mask, points=points, color=255)
    return mask

    # cv2.fillConvexPoly(img, points=points, color=255)
    # blur = scale(cv2.GaussianBlur(img,(55,55),0))
    # NOTE: THE FOLLOWING ARE FOR DEBUGGING
    # img = Image.fromarray(np.uint8(mask), 'L')
    # img.show()
    # return blur


def add_polygon(img, anno):
    """
    show anno
    A polygon standard coco-json format (x,y,x,y,x,y, etc.)
    anno:
    Index(['id', 'iscrowd', 'image_id',
    'category_id', 'segmentation', 'bbox',
    'area', 'category_name'],
    """
    for i, row in anno.iterrows():
        # segmentation
        seg = list(map(int, row.segmentation[0]))
        # number of points
        N_p = len(seg)
        # x's from list
        x, y = [], []
        # append x's and y's
        for idx, p in enumerate(seg):
            if idx%2==0:
                x.append(p)
            else:
                y.append(p)
        # add circle
        for k, v in zip(x, y):
            cv2.circle(img, (k,v), 20, (255,0,0),2)
    cv2.imshow('imgs', img)
    cv2.waitKey()

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

def load_json(fp):
    """
    Load json
    """
    with open(fp) as f:
        data = json.load(f)
        f.close()
    return(data)

if __name__ == '__main__':

    out_folder = '../MNG_annotations/annotations'
    img_source_folder = '../../dataset/cleanFrames/'

    # annotations stores
    MNG_GBM_anno = load_json('../MNG_annotations/labels_annotations-all-final_2022-02-23-09-43-09.json')
    # annotation map
    anno_map = {
        'meningioma': {
            'images': MNG_GBM_anno['images'],
            'anno': MNG_GBM_anno['annotations']
            }
    }
    img_folder = '../MNG_annotations/annotated_imgs/1/'
    for i, annotation in enumerate(anno_map['meningioma']['anno']):
        image_id = annotation['image_id']
        img_json = anno_map['meningioma']['images'][image_id - 1]
        # sanity check
        assert(img_json['id'] == image_id)
        
        img_height, img_width, file_name = img_json['height'], img_json['width'], img_json['file_name']

        # retrive the file
        ids = file_name[:-4][len('meningioma'):].split('_')
        if len(ids) == 2:
            patient_id, frame = ids
        else:
            patient_id = '0'
            frame = ids[0]

        #NOTE: COPY THE ANNOTATED IMAGES TO THE CORRECT DESTINATION, meningioma 0-frame13
        img_source_path = os.path.join(img_source_folder, 'meningioma {}'.format(patient_id), 'meningioma {}-frame{}.jpg'.format(patient_id, frame))
        # img_dst_path = os.path.join(img_folder, 'meningioma {}-frame{}.jpg'.format(patient_id, frame)) # with '.png'
        img_dst_path = os.path.join(img_folder, file_name) # with '.png'
        shutil.copy(img_source_path, img_dst_path)
        
        #NOTE: SAVE THE SEGMENTATION MASKS
        mask = make_mask(np.zeros((img_height, img_width)), annotation)
        save_path = os.path.join(out_folder, file_name[:-4]+' index{}'.format(i))
        np.save(save_path, mask)
