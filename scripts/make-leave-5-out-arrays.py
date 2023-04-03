import numpy as np
import os

### Preprocess data, add labels

#create a dictionary
actions = np.array(['cramped_synchronized_good', 'fidgety_movements'])
label_map = {label: num for num, label in enumerate(actions)}
DATA_PATH = r'C:\Users\Rahul Chawla\Desktop\Universiy\NVD\NVD-aloy\MP_6_coordinates_flattened'
DATA_PATH2 = r'C:\Users\Rahul Chawla\Desktop\Universiy\NVD\NVD-aloy'

print("actions = ", actions)

sequence_length = 250
sequences, labels = [], []

for action in actions:
    action_dir = os.path.join(DATA_PATH, action)
    i = 0
    for sequence_file in os.listdir(action_dir):
        i += 1
        sequence = []
        for frame_num in range(sequence_length):
            res_file = os.path.join(action_dir, sequence_file, str(frame_num) + '.npy')
            if os.path.exists(res_file):
                res = np.load(res_file)
                #print("res = ", res)
                sequence.append(res)
        if len(sequence) == sequence_length:
            sequences.append(sequence)
            labels.append(label_map[action])
        if i > 26:
            break

X = np.array(sequences)
y = np.array(labels)
#X = (X - X.mean(axis=1)) / X.std(axis=1)
folder = os.path.join(DATA_PATH2, '4Leave_5_Out_Iteration_all')
os.makedirs(folder, exist_ok=True)
np.save(os.path.join(folder, 'X.npy'), X)
np.save(os.path.join(folder, 'y.npy'), y)
print("done")
# Normalize the data
#
print("X_shape = ", X.shape)
 # Perform Leave-5-Out training 10 times
for i in range(10):
    test_indices = range(i*3, i*3+3)
    train_indices = [j for j in range(len(X)) if j not in test_indices]

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    # Create a new directory for this iteration
    iteration_folder = os.path.join(DATA_PATH2, '4Leave_5_Out_Iteration_{}'.format(i))
    os.makedirs(iteration_folder, exist_ok=True)

    np.save(os.path.join(iteration_folder, 'X_train.npy'), X_train)
    np.save(os.path.join(iteration_folder, 'X_test.npy'), X_test)
    np.save(os.path.join(iteration_folder, 'y_train.npy'), y_train)
    np.save(os.path.join(iteration_folder, 'y_test.npy'), y_test)
    print(i)

print("all done")
