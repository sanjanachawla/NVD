#code to populate folders with numpy arrays of coordinates extracted using mediapipe
# you should have prepared the folders first with another code

#to run this code:
#you need to change the path to the folder with videos of babies to run this code
#you need to change the name of the folder and subfolder that you want to populate with numpy arrays


#import cv2
#import mediapipe as mp
import numpy as np
import os

#name of high level folder that will be used for populating with numpy arrays (folde containing subfolders of action) (below uses a folder in the same root folder as this code)
#DATA_PATH = os.path.join('MP_data_all_coordinates_flattened')
#name of subfolder of action of baby (below uses a folder in the same root folder as this code)
action = 'cramped_synchronized'
#folder to get videos of babies (below uses a folder in the same root folder as this code)
np_folder = r'C:\Users\Rahul Chawla\Desktop\Universiy\NVD\NVD-aloy\MP_data_unflattened_11_seconds\cramped_synchronized'
NEW_DATA_PATH = r'C:\Users\Rahul Chawla\Desktop\Universiy\NVD\NVD-aloy\MP_6_coordinates_flattened\cramped_synchronized'
#NEW_DATA_PATH = os.path.join('MP_rel_coordinates_flattened')


np_num = 0
sequence = 0
for video in range(81):
    for frame in range(250):
        np_path =np_folder + '/' +str(video)+ '/' +str(frame) + '.npy'
        #print(np_path)
        data = np.load(np_path)
        rel_landmarks = np.array([data[15][0:3], data[16][0:3], data[13][0:3], data[14][0:3],data[25][0:3],data[26][0:3], data[27][0:3], data[28][0:3]]).flatten()
        #save that numpy array locally into the OS system
        new_npy_path = os.path.join(NEW_DATA_PATH,  str(video), str(frame))
        np.save(new_npy_path, rel_landmarks)


print("all good")