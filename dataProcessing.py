'''
Source code refer to https://www.thepythoncode.com/article/extract-frames-from-videos-in-python
'''

# from datetime import timedelta
import cv2
import numpy as np
import os
import csv
from PIL import Image

def format_timedelta(td):
    """Utility function to format timedelta objects in a cool way (e.g 00:00:20.05) 
    omitting microseconds and retaining milliseconds"""
    result = str(td)
    try:
        result, ms = result.split(".")
    except ValueError:
        return result + ".00".replace(":", "-")
    ms = int(ms)
    ms = round(ms / 1e4)
    return f"{result}.{ms:02}".replace(":", "-")


def get_saving_frames_durations(cap, saving_fps):
    """A function that returns the list of durations where to save the frames"""
    s = []
    # get the clip duration by dividing number of frames by the number of frames per second
    clip_duration = cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
    # use np.arange() to make floating-point steps
    for i in np.arange(0, clip_duration, 1 / saving_fps):
        s.append(i)
    return s

def ExtractFrames(video_file, frame_repo_path, frame_per_sec=100):
    filename, _ = os.path.splitext(video_file)
    
    # read the video file    
    cap = cv2.VideoCapture(video_file)
    # get the FPS of the video
    fps = cap.get(cv2.CAP_PROP_FPS)
    assert(fps != 0)
    # if the SAVING_FRAMES_PER_SECOND is above video FPS, then set it to FPS (as maximum)
    saving_frames_per_second = min(fps, frame_per_sec)
    # get the list of duration spots to save and [0::2] only keeps the 2nd element
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)[0::2]

    # start the loop
    count, amount_saved = 0, 0

    # create folder to save the frames of video
    patientID = filename.split('/')[-1]
    filename = frame_repo_path + '/' + patientID + '/' + 'framePerSec-{}-{}'.format(int(saving_frames_per_second), int(fps))
    # make a folder by the name of the video file
    if not os.path.isdir(filename):
        os.makedirs(filename) # create directory recursively

    while True:
        is_read, frame = cap.read()
        if not is_read:
            # break out of the loop if there are no frames to read
            break
        # get the duration by dividing the frame count by the FPS
        frame_duration = count / fps
        try:
            # get the earliest duration to save
            closest_duration = saving_frames_durations[0]
        except IndexError:
            # the list is empty, all duration frames were saved
            break
        if frame_duration >= closest_duration:
            # if closest duration is less than or equals the frame duration, 
            # then save the frame
            # frame_duration_formatted = format_timedelta(timedelta(seconds=frame_duration))
            cv2.imwrite(os.path.join(filename, "{}-frame{}.jpg".format(patientID, amount_saved)), frame) 
            amount_saved += 1

            # drop the duration spot from the list, since this duration spot is already saved
            try:
                saving_frames_durations.pop(0)
            except IndexError:
                pass
        # increment the frame count
        count += 1

    print("Save total of {}/{}".format(amount_saved, count))

def extractVideos(parent_dir, save_repo, save_frame_per_sec):
    videos = os.listdir(parent_dir)
    for video_name in videos:
        full_dir = parent_dir + video_name
        ExtractFrames(full_dir, save_repo, save_frame_per_sec)

def removeImagelogo(image):
    # Read a PIL image
    # image = Image.open(image_path)
    # numpy implementation to erase value
    img_tensor = np.array(image)

    # fixed location to erase the commercial logos for video GBM 1-16 and all Meingioma videos
    # img_tensor[303 : 326, 9 : 92, ...] = 0
    # img_tensor[300 : 333, 333: 397, ...] = 0

    # for GBM 17-18
    img_tensor[320 : 345, 9 : 92, ...] = 0
    img_tensor[315 : 348, 333: 397, ...] = 0
    return Image.fromarray(img_tensor)

def removeLogos(parent_dir, destination='../dataset/cleanFrames'):
    """Helper function to remove commercial logo from the video frames"""
    # iterate through each data point
    videos = os.listdir(parent_dir)
    for video in videos:

        # remove logos for all frames of a patient video
        patient_path = os.path.join(parent_dir, video)

        # skip non directory path
        if not os.path.isdir(patient_path):
            continue
        frame_rate_dirs = os.listdir(patient_path)

        # locate the framerate folder
        frames_path = os.path.join(patient_path, frame_rate_dirs[-1]) # NOTE: [0] with frame per sec 24/24
        # skip non directory path
        if not os.path.isdir(frames_path):
            continue
        frames = os.listdir(frames_path)

        for frame_name in frames:
            frame_path = os.path.join(frames_path, frame_name)
            frame = Image.open(frame_path)
            frame = removeImagelogo(frame)

            frame_dest = os.path.join(destination, video, frame_name)

            # make a folder by the name of the video file
            if not os.path.isdir(os.path.join(destination, video)):
                os.makedirs(os.path.join(destination, video)) # create directory recursively

            frame.save(frame_dest)

def output_annotations(output_dir='../', output_name='annotations.csv', input_dir='../cleanDistilledFrames'):
    '''
    Output a CSV file that contains name, annotation [0/1]
    where 0 = GBM and 1 = Meningeioma
    '''
    assert(os.path.isdir(input_dir))
    patient_dirs = os.listdir(input_dir)  
    dest = os.path.join(output_dir, output_name)
    
    with open(dest, 'x', encoding='UTF8') as f:

        # get a csv writer to write data
        writer = csv.writer(f)

        for patient in patient_dirs:
            # make sure it is a directory
            patient_dir = os.path.join(input_dir, patient)

            # skip non directory in the folder
            if not os.path.isdir(patient_dir):
                continue
            
            # open the patient directory
            patient_images = os.listdir(patient_dir)

            # write to csv fle
            for image_name in patient_images:
                if 'GBM' in image_name:
                    label = 0
                elif 'meningioma':
                    label = 1
                else:
                    label = -1
                
                # make the it is a valid label
                if label >= 0:
                    writer.writerow([image_name, label])

class CLE_Data_Handler:
    def __init__(self, source_loc='../cleanDistilledFrames'):
        assert(os.path.isdir(source_loc))

        self.patients = os.listdir(source_loc)

        # count how many patients are GBM and how many patients are meinigioma
        self.num_gbm = sum([1 for p in self.patients if 'GBM' in p])
        self.num_meningioma = sum([1 for p in self.patients if 'meningioma' in p])

        # 8:1:1 split => 14 2 2
        

    def split_data(self, num=3):
        '''
        if num = 3 => return train, val, test
        if num = 4 => return train, minival, val, test
        '''
        assert(num == 3 or num == 4)
        
        # generate index list
        gbm = [i for i in range(self.num_gbm)]
        m = [i for i in range(self.num_meningioma)]

        # train
        gbm_train, m_train = int(self.num_gbm * 0.8), int(self.num_meningioma * 0.8)
        
        # val 
        gbm_val, m_val = int(self.num_gbm * 0.1), int(self.num_meningioma * 0.1)

        # test
        gbm_test, m_test = self.num_gbm - gbm_train - gbm_val, self.num_meningioma - m_train - m_val

        # use numpy to help the randomness
        


    def write_annotations(self, data):
        pass

    def mean(self, data):
        pass

    def var(self, data):
        pass

if __name__ == '__main__':

    # # extract frame per GBM video
    # for i in range(1, 17):
    #     ExtractFrames('../dataset/GBM/GBM {}.mpg'.format(i), '../dataset/distilledFrames', frame_per_sec=100)


    # # extract frame per meningioma video
    # for i in range(0, 18):
    #     ExtractFrames('../dataset/meningioma/meningioma {}.mpg'.format(i), '../dataset/distilledFrames', frame_per_sec=100)

    # # remove logos in
    # removeLogos('../dataset/distilledFrames', '../dataset/cleanDistilledFrames')

    # output annotation
    output_annotations()