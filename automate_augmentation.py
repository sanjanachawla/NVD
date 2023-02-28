# Import everything needed to edit video clips
from moviepy.editor import *
rotation_angle=0
video_name = 0
number_of_videos =1
start_angle =0
max_rotation_angle = 90
rotation_increment = 20

for video_name in range (number_of_videos):
    for rotation_angle in range (start_angle,max_rotation_angle,rotation_increment):
        #loading video
        clip = VideoFileClip(str(video_name)+".mp4")
        #rotating the clip 45 degrees

        clip = clip.rotate(rotation_angle)
        # saving the clip
        clip.write_videofile("augmented_results/rotated_clip_"+str(rotation_angle)+".mp4")

    rotation_angle = 0

    for rotation_angle in range(-rotation_increment,-max_rotation_angle,-rotation_increment):
        # loading video
        clip = VideoFileClip(str(video_name) + ".mp4")
        # rotating the clip 45 degrees

        clip = clip.rotate(rotation_angle)
        # saving the clip
        clip.write_videofile("rotated_clip_" + str(rotation_angle) + ".mp4")

    #load videoclip
    clip = VideoFileClip(str(video_name) + ".mp4")
    # mirroring image according to the y axis
    clip = clip.fx(vfx.mirror_x)
    # saving the clip
    clip.write_videofile("mirrored_clip_" + str(rotation_angle) + ".mp4")

    for rotation_angle in range (start_angle,max_rotation_angle,rotation_increment):
        #loading video
        clip = VideoFileClip(str(video_name)+".mp4")

        # mirroring image according to the y axis
        clip = clip.fx(vfx.mirror_x)

        #rotating the clip
        clip = clip.rotate(rotation_angle)
        # saving the clip
        clip.write_videofile("mirrored_clip_" + str(rotation_angle) + ".mp4")

    rotation_angle = 0

    for rotation_angle in range(-rotation_increment,-max_rotation_angle,-rotation_increment):
        # loading video
        clip = VideoFileClip(str(video_name) + ".mp4")

        # mirroring image according to the y axis
        clip = clip.fx(vfx.mirror_x)

        # rotating the clip
        clip = clip.rotate(rotation_angle)
        # saving the clip
        clip.write_videofile("mirrored_clip_" + str(rotation_angle) + ".mp4")







