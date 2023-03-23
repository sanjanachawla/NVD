import math
from moviepy.editor import VideoFileClip
#path of video you desire to crop
path = r'C:\Users\aloys\Documents\Engineering\NVD\cerebral palsy\dataset\dance_videos\uncropped\scuba_diving_all.mp4'


print("please enter the start and end times that the cropped videos will have within the larger video")
#get user input for start time to crop
start_hour = int(input("start hour in video to crop: "))
start_min = int(input("start minute in video to crop: "))
start_sec = int(input("start second in video to crop: "))
#get user input for end time to crop
end_hour = int(input("end hour in video to crop: "))
end_min = int(input("end minute in video to crop: "))
end_sec = int(input("end second in video to crop: "))

#get user input for desired length of cropped videos
print("We can crop multiple smaller clips within the start and end time selected above")
print("please enter the desired length of the small clips in the format hour : minute : seconds ")
length_clip_hour = int(input("hour length (e.g. enter 1 if desired 1hr:20min:3sec clip: "))
length_clip_min = int(input("minute length (e.g. enter 20 if desired 1hr:20min:3sec clip: "))
length_clip_sec = int(input("second length (e.g. enter 3 if desired 1hr:20min:3sec clip: "))

#video_class = input("what is this video about? e.g. fidgety movements (to create a folder with this name): ")

number_of_clips = math.floor(((end_hour*3600+ end_min*60 + end_sec) - (start_hour*3600 + start_min*60 + start_sec))/(length_clip_hour*3600 + length_clip_min*60 + length_clip_sec))

print("number of clips is:" , number_of_clips)

# ---------------------------------above is a working code-------------------------------------

file = VideoFileClip(path)
i=0
for i in range (number_of_clips-1):
    new = file.subclip(t_start = (start_hour + i*length_clip_hour, start_min + i*length_clip_min, start_sec + i*length_clip_sec), t_end = (start_hour + (i+1)*length_clip_hour, start_min + (i+1)*length_clip_min, start_sec + (i+1)*length_clip_sec))
    new.write_videofile(r'C:\Users\aloys\Documents\Engineering\NVD\cerebral palsy\dataset\dance_videos\3_sec_under_the_sea/' + str(i) + '.mp4')