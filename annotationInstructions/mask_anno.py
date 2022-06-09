import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import pandas as pd
import json
import cv2
import numpy as np

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
    H, W = img.shape
    N = len(anno)
    n = 0
    for idx, row in anno.iterrows():
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
        # points for poly
        points = np.array(list(zip(x,y)))
        # fill convex hull
        cv2.fillConvexPoly(img, points=points, color=255)
        n+=1
    blur = scale(cv2.GaussianBlur(img,(55,55),0))
    return(blur)


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

def run(df, anno_map, out_folder):
    """
    load image apply annotations and create mask
    """
    # convert name of video and slice to annotation name
    df['anno_name'] = df['name_video'].apply(lambda x: x.replace(' ', '').strip())+'_'+df['n_slice'].astype(str)+'.png'
    for idx, row in df.iterrows():
        # out_file for dataloader
        save_path = os.path.join(out_folder, row.name_frame.replace(' ','')[:-3]+'npy')
        df.loc[idx, 'anno_filepath'] = save_path
        # if annotations exist in map
        if anno_map[row.label]:
            # get correct dataframes
            anno_dfs = anno_map[row.label]
            # images df
            images = anno_dfs['images']
            # annotation df
            anno = anno_dfs['anno']
            # file id for annotations
            file_id = images[images['file_name']==row.anno_name]
            if len(file_id)==0:
                df.loc[idx, 'bool_anno'] = 0
                mask = np.zeros((row.height, row.width))
            else:
                df.loc[idx, 'bool_anno'] = 1
                assert len(file_id['id'].values)==1
                file_id = file_id['id'].values[0]
                # get anno
                image_anno = anno[anno['image_id']==file_id]
                mask = make_mask(np.zeros((row.height, row.width)), image_anno)
        else:
            # if no annotation
            df.loc[idx, 'bool_anno'] = 0
            mask = np.zeros((row.height, row.width))
        # save anno
        np.save(save_path, mask)
    return(df)

if __name__ == '__main__':
    df_path = './input/ex-vivo-split/split.csv'
    out_folder = './input/ex-vivo-anno/'
    os.makedirs(out_folder)
    df = pd.read_csv(df_path)
    # annotations stores
    MNG_GBM_anno = load_json('/media/alfie/Storage/Clinical_Data/pCLE/cleopatra/ex-vivo/GBM_meningioma_annotations/labels_annotations-all-final_2022-02-23-09-43-09.json')
    ductal_anno = load_json('/media/alfie/Storage/Clinical_Data/pCLE/cleopatra/ex-vivo/ductal_annotations/labels_ductal-metastasis-_2022-03-28-01-06-30.json')
    # annotation map
    anno_map = {
        'meningioma': {
            'images': pd.DataFrame(MNG_GBM_anno['images']),
            'anno': pd.DataFrame(MNG_GBM_anno['annotations'])
            },
        'ductal_metastasis': {
            'images': pd.DataFrame(ductal_anno['images']),
            'anno': pd.DataFrame(ductal_anno['annotations'])
            },
        'glioblastoma': None
    }
    # run
    df = run(df, anno_map, out_folder)
    # save df
    df.to_csv(df_path, index=False)
