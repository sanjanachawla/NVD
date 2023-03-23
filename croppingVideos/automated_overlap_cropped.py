import math
from moviepy.editor import VideoFileClip
#path of video you desire to crop
path = r'C:\Users\aloys\Documents\Engineering\NVD\cerebral palsy\dataset\uncropped_videos\GMA_1hr.mp4'

video_categories = ['abnormal_fidgety', 'abnormal_general', 'chaotic', 'cramped_synchronized', 'fidgety_movements', 'normal_preterm','postterm_writhing']
#get user input for start time to crop
start_hour = [0,0,0,0,0,0,0]
start_min = [42,28,40,35,20,4,10]
start_sec = [41,15,28,14,3,48,39]
#get user input for end time to crop
end_hour = [0,0,0,0,0,0,0]
end_min = [50,34,41,40,27,9,18]
end_sec = [23,37,45,2,35,42,34]



#get user input for desired length of cropped videos
print("We can crop multiple smaller clips within the start and end time selected above")
print("please enter the desired length of the small clips in the format hour : minute : seconds ")
length_clip_hour = int(input("hour length (e.g. enter 1 if desired 1hr:20min:3sec clip: "))
length_clip_min = int(input("minute length (e.g. enter 20 if desired 1hr:20min:3sec clip: "))
length_clip_sec = int(input("second length (e.g. enter 3 if desired 1hr:20min:3sec clip: "))


file = VideoFileClip(path)

j = 0
for j in range (7):
    number_of_clips = 2 * math.floor(((end_hour[j] * 3600 + end_min[j] * 60 + end_sec[j]) - (start_hour[j] * 3600 + start_min[j] * 60 + start_sec[j])) / (length_clip_hour * 3600 + length_clip_min * 60 + length_clip_sec))

    print("number of clips is:", number_of_clips)
    i=0
    for i in range (number_of_clips):
        new = file.subclip(t_start = (start_hour[j] + i*length_clip_hour/2, start_min[j] + i*length_clip_min/2, start_sec[j] + i*length_clip_sec/2), t_end = (start_hour[j] + (i*length_clip_hour+2*length_clip_hour)/2, start_min[j] + (i*length_clip_min+2*length_clip_min)/2, start_sec[j] + (i*length_clip_sec+2*length_clip_sec)/2))
        new.write_videofile(r'C:\Users\aloys\Documents\Engineering\NVD\cerebral palsy\dataset\overlapped_11_sec_clips'+'/' +video_categories[j] +'/' + str(i) + '.mp4')

    j = j+1
