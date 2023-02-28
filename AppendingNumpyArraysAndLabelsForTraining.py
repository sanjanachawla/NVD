import numpy as np
import os


### Preprocess data, add labels

actions = np.array(['fidgety','cramped_synchronized'])

#from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical

#create a dictionary
label_map = {label:num for num, label in enumerate(actions)}

DATA_PATH = os.path.join('MP_data')
actions = np.array(['cramped_synchronized','fidgety'])

sequence_length = 120
sequences, labels = [], []
no_sequences = 53 
action = 'cramped_synchronized'
for sequence in range(no_sequences):
  window =[]
  for frame_num in range (sequence_length):
    res = np.load(os.path.join(DATA_PATH, action, str(sequence), str(frame_num) + '.npy'))
    window.append(res)
  sequences.append(window)
  labels.append(label_map[action])

no_sequences = 53
action = 'fidgety'
for sequence in range(no_sequences):
  window =[]
  for frame_num in range (sequence_length):
    res = np.load(os.path.join(DATA_PATH, action, str(sequence), str(frame_num) + '.npy'))
    window.append(res)
  sequences.append(window)
  labels.append(label_map[action])

X = np.array(sequences)

npy_path = os.path.join('X')
np.save(npy_path, X)
npy_path = os.path.join('labels')
np.save(npy_path, np.array(labels))

