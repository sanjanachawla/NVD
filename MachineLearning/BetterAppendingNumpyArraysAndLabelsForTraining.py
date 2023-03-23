import numpy as np
import os


### Preprocess data, add labels

actions = np.array(['CP','noCP'])

#from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical

#create a dictionary
label_map = {label:num for num, label in enumerate(actions)}

DATA_PATH = os.path.join('Edmund_data_changed')

sequence_length = 998
sequences, labels = [], []


array_folder = r'C:\Users\aloys\Documents\Engineering\NVD\cerebral palsy\code\MachineLearning\Edmund_data_changed'

action = 'CP'
os.chdir(array_folder + '/'+ action)
for file_name in os.listdir():
  window =[]
  os.chdir(array_folder + '/' + action + '/' + file_name)
  print(file_name)
  for frame_num in os.listdir():
    res = np.load(os.path.join(array_folder, action, str(file_name), str(frame_num)))
    window.append(res)
  sequences.append(window)
  labels.append(label_map[action])


action = 'noCP'
os.chdir(array_folder + '/' + action)
for file_name in os.listdir():
  window =[]
  os.chdir(array_folder + '/' + action + '/' + file_name)
  for frame_num in os.listdir():
    res = np.load(os.path.join(array_folder, action, str(file_name), str(frame_num)))
    window.append(res)
  sequences.append(window)
  labels.append(label_map[action])

print(sequences)
X = np.array(sequences)

npy_path = r'C:\Users\aloys\Documents\Engineering\NVD\cerebral palsy\code\MachineLearning\X'
np.save(npy_path, X)
#npy_path = os.path.join('labels')
npy_path = r'C:\Users\aloys\Documents\Engineering\NVD\cerebral palsy\code\MachineLearning\labels'
np.save(npy_path, np.array(labels))

