'''
Source code refer to https://www.thepythoncode.com/article/extract-frames-from-videos-in-python
'''

# from datetime import timedelta
import cv2
import numpy as np
import os

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
    # get the list of duration spots to save
    saving_frames_durations = get_saving_frames_durations(cap, saving_frames_per_second)
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
    # fixed location to erase the commercial logos
    img_tensor[303 : 326, 9 : 92, ...] = 0
    img_tensor[300 : 333, 333: 397, ...] = 0

    return Image.fromarray(img_tensor)

def removeLogos(parent_dir, destination='../dataset/cleanFrames'):
    """Helper function to remove commercial logo from the video frames"""
    # filter_list = ['meningioma ', 'GBM']

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
        frames_path = os.path.join(patient_path, frame_rate_dirs[0]) # NOTE: [0] with frame per sec 24/24
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


if __name__ == '__main__':
    # save all frames

    # extract frame per video for meningioma tumor
    # ExtractFrames('../dataset/meningioma/meningioma 18.mpg', '../dataset/Frames', frame_per_sec=100)
    # removeLogos('../dataset/Frames/meningioma 0/framePerSec-24-24/meningioma 0-frame10.jpg')


    # remove logos in
    removeLogos('../dataset/Frames')



## pytorch implementation to remove logos
    # Define a transform to convert PIL 
    # image to a Torch tensor
    # tensorToPIL = transforms.ToPILImage()
    # toTensor = transforms.ToTensor()
    
    # transform = transforms.PILToTensor()
    # Convert the PIL image to Torch tensor
    # img_tensor = toTensor(image)
    # img_tensor[..., 303 : 326, 9 : 92] = 0
    # img_tensor[..., 300 : 333, 333: 397] = 0

    # convert the tensor to PIL image using above transform
    # img = tensorToPIL(img_tensor)