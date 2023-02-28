#code to rename files in a folder as 0.mp4,1.mp4,2.mp4,3.mp4,4.mp4,...
#this is useful for standardizing the names for training ML model
import os

video_folder = r'C:\Users\aloys\Documents\Engineering\NVD\cerebral palsy\code\PoseEstimation\5_sec_clips\cramped_synchronized'
os.chdir(video_folder)

i=0
for file_name in os.listdir():

    os.rename(file_name, str(i))
    i = i+1
i=0
for file_name in os.listdir():

    os.rename(file_name, str(i) + '.mp4')
    i = i+1