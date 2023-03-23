Workflow that I used for training the LSTM model using the codes I put on GitHub:

1) Download this GMA training video https://www.youtube.com/watch?v=4c4oUJSw3rQ&t=22s using a downloader e.g. https://wave.video/ca/convert/youtube-to-mp4
2) Make separate folders for storing 5 second segments of the 1 hour video. Name the folders based on the categories of the 1 hour video: abnormal fidgety movements, abnormal general movements, etc.
3) Take note of the start time and end time (min and sec)  of the different types of movement in the 1 hour video
4) Crop the 1 hour video GMA training into 5 second segments -- use the code "cropping videos" or the "overlap cropped" for that; input the start and end time for each category of movement from step 3 into the code
5) Go through the 5 second videos and manually delete the videos that have no baby movement  (e.g. video transitions with no babies in frame, baby video frozen)
6) Use the rename_files code to rename all the 5 second segments in each folder as 0,1,2,3,4... to clean the gaps in the sequence of names after deletions
7) Run the "CreateFoldersForCollection" to create hundreds of organized folders for collecting data in numpy arrays
8) Run the "collect_keypoints_and_save" to detect the pose in all 5 second videos in one category of movement and save the results in the folders you created in step 7
9) Run the "AppendingNumpyArraysAndLabelsForTraining" to append all numpy arrays with labels to train the LSTM model
10) Copy and paste the X and y numpy arrays created in step 11 into the sample_data folder in this Google Collab to train the LSTM learning model. Only run the code under "Build and Train LSTM Neural Network" code in Collab