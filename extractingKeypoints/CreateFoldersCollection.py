#this code creates folders with categories of actions that we want to detect
# each baby video will have one folder for it named 1,2,3,4,5,6,... no_sequences
#to run this code:
# change the actions array
# change the name of the data path that you want to create
# change the number of videos that you want to detect (each video will have its own folder)

import numpy as np
import os

#path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_data')

#actions that we try to detect
actions = np.array(['abnormal_fidgety', 'abnormal_general', 'chaotic', 'cramped_synchronized', 'fidgety_movements', 'normal_preterm','postterm_writhing'])

#number of videos ?????????????????????????????????????????????????????????????????????????????????????????????????????????????????
no_sequences = 158

for action in actions:
  for sequence in range (no_sequences):
    try:
      os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
    except:
      pass
