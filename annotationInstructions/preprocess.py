import numpy as np
import tqdm
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import seaborn as sns

def load_video(path, n_f):
    """
    Load video using cv2.VideoCapture
    stores frames in list
    return list
    """
    # load capture
    cap = cv2.VideoCapture(path)
    # define list
    frames = []
    # open cap and store franes
    if not cap.isOpened():
        print("Cannot open video")
        exit()
    i=0
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame {}, skipping frame ...".format(i))
            break
        # Our operations on the frame come here
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = frame[y1:y2, x1:x2]
        frames.append(frame)
        # stop at number of frames
        i+=1
        if i==n_f:
            break
    # fps
    framespersecond= int(cap.get(cv2.CAP_PROP_FPS))
    # When everything done, release the capture
    cap.release()
    #cv2.destroyAllWindows()
    return(np.array(frames), framespersecond)

def break_into_frames(in_folder, out_folder, out_depth=1):
    """
    split frames based on video path
    """
    # open fileptaths
    videos_filepaths = glob.glob(os.path.join(in_folder,'*.mpg'))
    # init df
    print('Reading files in folder...')
    f_dict = []
    for fp in tqdm.tqdm(videos_filepaths):
        # grab data from video
        fp_name = os.path.splitext(os.path.basename(fp))[0]
        frames, fps = load_video(fp, -1)
        # get 
        depth, h, w = frames.shape
        for i in range(depth):
            img = frames[i, :, :]
            assert img.shape[0]==h
            f_name = fp_name+'_{}.npy'.format(i)
            f_filepath = os.path.join(out_folder, f_name).replace(' ', '')
            np.save(f_filepath, img)
            #plt.imsave(f_filepath,img, cmap='gray')
            f_dict.append(
                {'filepath_frame': f_filepath,
                'name_frame': f_name,
                'name_video': fp_name,
                'n_slice': int(i),
                'height': int(h),
                'width': int(w)},
                )

    df = pd.DataFrame(f_dict, columns=['filepath_frame', 'name_frame', 'name_video', 'n_slice', 'height', 'width'])
    return(df)


def get_dmt(in_folder, out_folder):
    # crop
    x1, y1 = 92, 84
    x2, y2 = 485, 490
    print('Reading DMT files...')
    # folder map
    file_map = {
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8
    }
    file_paths = os.path.join(in_folder,'*/*.pkl')
    files = glob.glob(file_paths)
    f_dict = []
    for f in files:
        # stats
        split_filepath = f.split('/')
        name_frame = split_filepath[-1]
        name_video = file_map[split_filepath[-2]]
        n_slice = name_frame.split('.')[0]
        # mask
        x = np.load(f, allow_pickle=True)
        # remove nan values
        x = np.where(np.isnan(x), 0, x)
        # remove negative values
        x = np.where(x>0, x, 0)
        # crop 
        x1, y1 = 92, 84
        x2, y2 = 485, 493
        crop = x[y1:y2, x1:x2]
        # min max scale to 255 of crop 
        max_ = crop.max()
        min_ = crop.min()
        x = (x-min_)/((max_-min_)+1e-7)
        x*=255
        h,w = x.shape
        f_filepath = os.path.join(out_folder, split_filepath[-2]+'_'+name_frame).replace(' ', '').replace('pkl', 'png')
        # np.save(f_filepath, x)
        plt.imsave(f_filepath, x, cmap='gray')
        f_dict.append(
            {'filepath_frame': f_filepath,
            'name_frame': name_frame,
            'name_video': name_video,
            'n_slice': int(n_slice),
            'height': int(h),
            'width': int(w)},
            )
    df = pd.DataFrame(f_dict, columns=['filepath_frame', 'name_frame', 'name_video', 'n_slice', 'height', 'width'])
    return(df)


def assign_folds(df):
    """
    Split by video rather than random split
    """
    # assign folds
    df['fold'] = 0
    # for each label
    for c in pd.unique(df.label):
        # split by 75%
        unique_videos = pd.unique(df[df.label == c]['name_video'])
        split_point = round(len(unique_videos)*0.75)
        # training videos
        train_videos = unique_videos[:split_point]
        # valid videos
        valid_videos = unique_videos[split_point:]
        # assign train
        for tv in train_videos:
            df.loc[(df.name_video==tv), 'fold'] = 0
        # assign valid
        for vv in valid_videos:
            df.loc[(df.name_video==vv), 'fold'] = 1
    return(df)


def run(out_depth=1):
    # get and make dfs
    base = '/media/alfie/Storage/Clinical_Data/pCLE/cleopatra/ex-vivo/'
    gbm_folder = base + 'GBM/'
    mng_folder = base + 'meningioma/'
    out_folder = base+'processed/'
    # make out dir
    os.makedirs(out_folder,exist_ok=True)
    # get_dfs
    gbm_df = break_into_frames(gbm_folder, out_folder, out_depth=out_depth)
    mng_df = break_into_frames(mng_folder, out_folder, out_depth=out_depth)
    """
    For reference,
    {
        MNG = 0
        GBM = 1
        DMT = 2
    }
    """
    # assign labels    
    mng_df['label'] = ['meningioma']*len(mng_df)
    mng_df['n_label'] = [0]*len(mng_df)
    gbm_df['label'] = ['glioblastoma']*len(gbm_df)
    gbm_df['n_label'] = [1]*len(gbm_df)
    # concat dfs
    df = pd.concat([gbm_df, mng_df]).reset_index(drop=True)
    # set index
    df['ID'] = df.index
    # assign folds
    df = assign_folds(df)
    df.to_csv(os.path.join(out_folder, 'split.csv'), index=False)
    return(df)

if __name__ == '__main__':
    df = run()
    sns.countplot(data=df, x='name_video', hue='fold')
    plt.show()