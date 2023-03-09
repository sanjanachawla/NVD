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
FLATTENED_DATA_PATH = os.path.join('MP_data_flattened_11_seconds')
UNFLATTENED_DATA_PATH = os.path.join('MP_data_unflattened_11_seconds')

#name of subfolder of action of baby (below uses a folder in the same root folder as this code)
actions = np.array(['abnormal_fidgety', 'abnormal_general', 'chaotic', 'cramped_synchronized', 'fidgety_movements', 'normal_preterm','postterm_writhing'])

#folder to get videos of babies (below uses a folder in the same root folder as this code)
video_folder = r'C:\Users\aloys\Documents\Engineering\NVD\cerebral palsy\dataset\overlapped_11_sec_clips'

#video_num = 0
#sequence = 0

offset = 25 #number of pose frames to skip in beginning of video before starting to save pose data

for action in actions:
    for sequence in range(158):
        video_path = video_folder + '/' +action+ '/' + str(sequence) + '.mp4'
        cap = cv2.VideoCapture(video_path)
        #loop through all frames of video
        frame_num=0

        img_array = []

        while frame_num <275:
            #detect keypoints of pose using mediapipe
            success, img = cap.read()  # read a frame
            if (success == False):
                frame_num = frame_num+1
                break
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to RGB
            results = pose.process(imgRGB)


            #save keypoints into a numpy array starting after one second of video
            if (frame_num > offset-1):
                if (results.pose_landmarks):
                    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

                    cv2.putText(img, 'l w ' + str(round(results.pose_landmarks.landmark[15].visibility, 2)), (40, 50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
                    cv2.putText(img, 'r w ' + str(round(results.pose_landmarks.landmark[16].visibility,2)), (40, 80), cv2.FONT_HERSHEY_PLAIN,2, (255, 0, 0), 3)
                    cv2.putText(img, 'l f ' + str(round(results.pose_landmarks.landmark[31].visibility,2)), (40, 110), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
                    cv2.putText(img, 'r f ' + str(round(results.pose_landmarks.landmark[32].visibility,2)), (40,140), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)

                keypoints_flattened = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(132)
                keypoints_unflattened = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 4))


                #save the flattened numpy array locally into the OS system
                npy_path = os.path.join(FLATTENED_DATA_PATH, action, str(sequence), str(frame_num-offset))
                np.save(npy_path, keypoints_flattened)

                # save the unflattened numpy array locally into the OS system
                npy_path = os.path.join(UNFLATTENED_DATA_PATH, action, str(sequence), str(frame_num - offset))
                np.save(npy_path, keypoints_unflattened)

                img_array.append(img)

            frame_num = frame_num+1

        if (success == False):
            break

        height, width, layers = img.shape
        size = (width, height)

        out = cv2.VideoWriter('overlayed_11_seconds'+ '/' + action+'/' +str(sequence)+'.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 25, size)

        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()

        #sequence = sequence + 1


