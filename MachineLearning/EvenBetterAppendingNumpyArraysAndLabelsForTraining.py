import numpy as np
import os


### Preprocess data, add labels

actions = np.array(['3_sec_gagnam_style', '3_sec_under_the_sea'])

#from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical

#create a dictionary
label_map = {label:num for num, label in enumerate(actions)}

#DATA_PATH = os.path.join('Edmund_datas')

sequence_length = 998
sequences, labels = [], []


array_folder = r'C:\Users\aloys\Documents\Engineering\NVD\cerebral palsy\code\PoseEstimation\dance_MP_data_flattened_3_seconds'
num_frames = 88
action = '3_sec_gagnam_style'
os.chdir(array_folder + '/'+ action)
num_videos = 14

for file_name in os.listdir():
  if(int(file_name) > num_videos):
    break
  os.chdir(array_folder + '/' + action + '/' + file_name)
  print(file_name)
  window =[]
  i=0

  for frame_num in range(num_frames):
    res = np.load(os.path.join(array_folder, action, str(file_name), str(frame_num) +'.npy'))
    window.append(res)
    print(frame_num)
    if((i%num_frames == 0)and(i>0)):
      print("i = ", i)
      print("appended")
      print(window)
      sequences.append(window)
      labels.append(label_map[action])
      window = []
    i = i + 1

action = '3_sec_under_the_sea'
os.chdir(array_folder + '/' + action)
num_frames = 88

for file_name in os.listdir():
  os.chdir(array_folder + '/' + action + '/' + file_name)
  window =[]
  i=0

  for frame_num in range(num_frames):
    res = np.load(os.path.join(array_folder, action, str(file_name), str(frame_num)+'.npy'))
    window.append(res)
    if ((i % 110 == 0) and (i > 0)):
      sequences.append(window)
      print(labels)
      labels.append(label_map[action])
      window = []
    i = i + 1

print (sequences)
print(sequences.shape)
X = np.array(sequences)
print(X.shape)

npy_path = r'C:\Users\aloys\Documents\Engineering\NVD\cerebral palsy\code\MachineLearning\X'
np.save(npy_path, X)
#npy_path = os.path.join('labels')
npy_path = r'C:\Users\aloys\Documents\Engineering\NVD\cerebral palsy\code\MachineLearning\labels'
np.save(npy_path, np.array(labels))

