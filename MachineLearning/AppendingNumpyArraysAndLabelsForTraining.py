import numpy as np
import os


### Preprocess data, add labels

actions = np.array(['3_sec_gagnam_style', '3_sec_under_the_sea'])

#from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical

#create a dictionary
label_map = {label:num for num, label in enumerate(actions)}

#DATA_PATH = os.path.join('Curated_MP_data')

sequence_length = 88
sequences, labels = [], []
no_sequences = 60
action = '3_sec_gagnam_style'

array_folder = r'C:\Users\aloys\Documents\Engineering\NVD\cerebral palsy\code\PoseEstimation\dance_MP_data_flattened_3_seconds'

for sequence in range(no_sequences):
  window =[]
  for frame_num in range (sequence_length):
    res = np.load(os.path.join(array_folder, action, str(sequence), str(frame_num) + '.npy'))
    window.append(res)
  sequences.append(window)
  labels.append(label_map[action])

no_sequences = 60
action = '3_sec_under_the_sea'
for sequence in range(no_sequences):
  window =[]
  for frame_num in range (sequence_length):
    res = np.load(os.path.join(array_folder, action, str(sequence), str(frame_num) + '.npy'))
    window.append(res)
  sequences.append(window)
  labels.append(label_map[action])

X = np.array(sequences)

npy_path = os.path.join('X')
np.save(npy_path, X)
npy_path = os.path.join('labels')
np.save(npy_path, np.array(labels))

