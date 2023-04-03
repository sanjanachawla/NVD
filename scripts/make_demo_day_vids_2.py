#What we want:
# Videos to loop through with the prediction overlayed on it
# run on Google Collab, make array of test predictions
# - then save in array to download.
# upload the arrays here
# overlap 5 vids with the test arrays at each time.
# load Videos

import numpy as np
import os

import cv2
import mediapipe as mp
import os

mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.utils import to_categorical
print("done imports")

actions = np.array(['fidgety_movements','cramped_synchronized'])
#load model that we trained
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(250, 24)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
print("instanstiated model")
DATA_PATH3 = r'C:\Users\Rahul Chawla\Desktop\Universiy\NVD\NVD-aloy'
model.load_weights(os.path.join(DATA_PATH3+'/models/leave-5-out-9_5_100_100 (1).h5'))
print("loaded weights")


### Preprocess data, add labels

#create a dictionary
label_map = {label: num for num, label in enumerate(actions)}
DATA_PATH = r'C:\Users\Rahul Chawla\Desktop\Universiy\NVD\NVD-aloy\MP_6_coordinates_flattened'
DATA_PATH2 = r'C:\Users\Rahul Chawla\Desktop\Universiy\NVD\NVD-aloy\vidsblurred\vidsblurred_good'

print("actions = ", actions)

class poseDetector():
    def __init__(self, mode = False, model_complexity = 1, smooth_landmarks = True, enable_segmentation = False, smooth_segmentation = True,
                  detectionCon = 0.9, trackCon = 0.5):

        self.mode = mode #whenever we create a new object it will have it's own variables
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(self.mode, self.model_complexity, self.smooth_landmarks, self.enable_segmentation, self.smooth_segmentation, self.detectionCon, self.trackCon)



    def findPose(self, img, draw = True ): # will ask user if we want to draw or not
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.pose.process(imgRGB)
        if results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, results.pose_landmarks,
                                       self.mpPose.POSE_CONNECTIONS)
        return img, results


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros(33 * 4)
    return pose
mpDraw = mp.solutions.drawing_utils
mpPose = mp.solutions.pose
pose = mpPose.Pose()

#set mediapipe model
detector = poseDetector()

#folder to get videos of babies (below uses a folder in the same root folder as this code)


for action in actions:
    #action = 'cramped_synchronized'
    i = 0
    action_dir = os.path.join(DATA_PATH2, action)
    print(os.listdir(action_dir))
    correct = 0
    number = 0
    for vid in os.listdir(action_dir):
        vid_path = os.path.join(DATA_PATH2,action, vid)
        print("vid = ", vid)
        print("vidpath = ", vid_path)
        cap = cv2.VideoCapture(vid_path)
        #cv2.namedWindow("image", cv2.WINDOW_NORMAL)
        #cv2.setWindowProperty("image", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        frame_num=0
        X_array = np.zeros((250, 24))
        sequence = []
        while frame_num <275:
            #detect keypoints of pose using mediapipe
            success, img = cap.read()  # read a frame
            imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to RGB
            results = pose.process(imgRGB)

            #display_results!!!!!!!!!!!!
            if results.pose_landmarks:
                mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                # for id, lm in enumerate(results.pose_landmarks.landmark):
            cv2.imshow("image", img)
            # Delay until next frame is shown
            cv2.waitKey(1)

            #save keypoints into a numpy array
            keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33,4))
            #print(keypoints)
            rel_landmarks = np.array(
                [keypoints[15][0:3], keypoints[16][0:3], keypoints[13][0:3], keypoints[14][0:3], keypoints[25][0:3], keypoints[26][0:3],
                 keypoints[27][0:3], keypoints[28][0:3]]).flatten()
            #save that numpy array locally into the OS system
            #print("rel landmarks = ", rel_landmarks)
        #    X_array[frame_num][:] = rel_landmarks
            if(frame_num>24):
                sequence.insert(0, rel_landmarks)
            #print("sequence = ", sequence)
            frame_num = frame_num+1
            #print("X_array =", X_array)

        #print("X_array = ", X_array)
        # We now want to feed this video into the model
        # Normalize the data
        sequence = sequence[:250]  # we take 120 frames since we trained with 120 frames of data
        # print (sequence)
        if len(sequence) == 250:
            test = np.expand_dims(sequence, axis=0)
            test_ins = (test-test.mean(axis=1))/test.std(axis=1)
            #print("test.shape", test.shape)
            res = model.predict(test_ins)[0]
            print(res)
            print(actions[np.argmax(res)])

            cap = cv2.VideoCapture(vid_path)
            frame_num = 0
            X_array = np.zeros((250, 24))
            sequence = []
            if(actions[np.argmax(res)] == action):
                correct = correct+1
                if(action == 'crampled_synchronized'):
                    TP = TP+1 #true positive
                else:
                    TN = TN+1 #true negative
            else:
                if (action == 'crampled_synchronized'):
                    FP = FP + 1  # false positive
                else:
                    FN = FN + 1  # false negative

            number = number+1
            print("correct = ", correct)
            print("number =  ", number)
            print("accuracy = ", correct/number)
            while frame_num < 275:
                # detect keypoints of pose using mediapipe
                success, img = cap.read()  # read a frame
                imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert image to RGB
                results = pose.process(imgRGB)

                # display_results!!!!!!!!!!!!
                if results.pose_landmarks:
                    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
                    # for id, lm in enumerate(results.pose_landmarks.landmark):

                cv2.imshow("image", img)
                # Delay until next frame is shown
                cv2.waitKey(1)

                # save keypoints into a numpy array
                keypoints = np.array([[res.x, res.y, res.z, res.visibility] for res in
                                      results.pose_landmarks.landmark]) if results.pose_landmarks else np.zeros((33, 4))
                # print(keypoints)
                # save that numpy array locally into the OS system
                # print("rel landmarks = ", rel_landmarks)
                #    X_array[frame_num][:] = rel_landmarks
                # print("sequence = ", sequence)
                frame_num = frame_num + 1
                if(actions[np.argmax(res)] =='fidgety_movements'):
                    prediction = "No CP signs"
                else:
                    prediction = "CP cigns"
                if(action == "fidgety_movements"):
                    true_val = "No CP Signs"
                else:
                    true_val = "CP Signs"



                # cv2.putText(img, "pred = "+actions[np.argmax(res)], (50, 350), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
                # cv2.putText(img, "true = "+action, (50, 375), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
                # cv2.putText(img, "conf = "+str(np.round(np.max(res), 3)), (50, 400), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
                # cv2.imshow("image", img)
                # cv2.waitKey(1)
                cv2.putText(img, "pred = " + prediction, (50, 350), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0),
                            3)
                cv2.putText(img, "true = " + true_val, (50, 375), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 3)
                cv2.putText(img, "conf = " + str(np.round(np.max(res), 3)), (50, 400), cv2.FONT_HERSHEY_PLAIN, 2,
                            (255, 0, 0), 3)
                cv2.imshow("image", img)
                cv2.waitKey(1)


        i = i+1
        # if i>26:
        #     break

        #print(y_prediction)
        #print(actions[np.argmax(y_prediction)])


