#code to populate folders with numpy arrays of coordinates extracted using mediapipe
# you should have prepared the folders first with another code

#to run this code:
#you need to change the path to the folder with videos of babies to run this code
#you need to change the name of the folder and subfolder that you want to populate with numpy arrays


import cv2
import mediapipe as mp
import numpy as np
import os

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

#name of high level folder that will be used for populating with numpy arrays (folde containing subfolders of action) (below uses a folder in the same root folder as this code)
DATA_PATH = os.path.join('MP_data')

#name of subfolder of action of baby (below uses a folder in the same root folder as this code)
action = 'spasm'

#folder to get videos of babies (below uses a folder in the same root folder as this code)
#video_folder = os.path.join('augmented_results')
video_folder = r'C:\Users\aloys\Documents\Engineering\NVD\cerebral palsy\dataset\uncropped_videos'

video_num = 0
sequence = 0
max_num_of_videos = 1
num_frames_in_video = 370

video_path = video_folder + '/' + 'Signs of Infantile Spasms part 1'+ '.mp4'

for video_num in range(max_num_of_videos):
    #video_path = video_folder + '/' +str(sequence) + '.mp4'

    cap = cv2.VideoCapture(video_path)
    #loop through all frames of video
    frame_num=0

    img_array = []

    while frame_num <num_frames_in_video:
        #detect keypoints of pose using mediapipe
        success, img = cap.read()  # read a frame
        if (success == False):
            frame_num = frame_num + 1
            break
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to RGB
        results = pose.process(imgRGB)

        #display_results!!!!!!!!!!!!
        if (results.pose_landmarks and frame_num >25):
            mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
            # for id, lm in enumerate(results.pose_landmarks.landmark):
        cv2.imshow("image", img)
        # Delay until next frame is shown
        cv2.waitKey(1)

        #save keypoints into a numpy array starting after one second of video
        if (frame_num >25):
            keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)

            #save that numpy array locally into the OS system
            npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
            np.save(npy_path, keypoints)

            img_array.append(img)

            height, width, layers = img.shape
            size = (width, height)

        frame_num = frame_num+1

    out = cv2.VideoWriter('overlayed'+str(video_num)+'.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30, size)

    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()



