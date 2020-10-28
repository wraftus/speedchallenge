import numpy as np
import ast
import random

from tensorflow.keras import layers, models, losses
from tensorflow import data

import matplotlib.pyplot as plt

# wrapper class to create dataset form features text file
FEATS_PER_EXAMPLE=100
EXAMPLES_PER_FRAME=75
class Dataset:
  def __init__(self, dataset_file, features_file=None, speed_file=None):
    if bool(features_file) != bool(speed_file):
      raise Exception("You only have one of features_file or speed_file")

    # if features_file is not None, we need to construct the dataset
    if features_file:
      self.X = []
      self.y = []

      # read in speeds from each frame
      with open(speed_file) as raw_speed:
        speeds = raw_speed.read().splitlines()
      speeds = [float(speed.rstrip()) for speed in speeds[1:]]

      # read in raw feature info into numpy array
      with open(features_file) as raw_feats:
        num_frame = 0
        while True:
          if (num_feats := raw_feats.readline().rstrip()):
            old_feats = []
            new_feats = []

            # get all features from the new/prev frame
            for i in range(int(num_feats)):
              # only take the first 100 matching features per frame
              feat_pair = ast.literal_eval(raw_feats.readline().rstrip())
              new_feats.append(feat_pair[0] + feat_pair[2])
              old_feats.append(feat_pair[1] + feat_pair[3])

            # create examples from feature pairs
            for i in range(EXAMPLES_PER_FRAME):
              rand_idx = random.sample(range(len(new_feats)), FEATS_PER_EXAMPLE)
              shuffled_new_feats = [new_feats[idx] for idx in rand_idx]
              shuffled_old_feats = [old_feats[idx] for idx in rand_idx]
              self.X.append([shuffled_new_feats, shuffled_old_feats])
              self.y.append(speeds[num_frame])
            print("\rgot examples from frame {}".format(num_frame := num_frame + 1), end='')    

          # break when we run out of frames      
          else:
            print()
            break

      # turn out X into a numpy array (# frames, # of feats, size of feat + 2, 2)
      self.X = np.array(self.X, np.uint16)
      self.X = np.swapaxes(self.X, 1, 3)
      self.y = np.array(self.y)
      print("X shape {}".format(self.X.shape))
      print("y shape{}".format(self.y.shape))

      # save numpy arrays
      np.savez(dataset_file, self.X, self.y)

    #else we read the dataset from the file
    else:
      from_file = np.load("{}.npz".format(dataset_file))
      self.X, self.y = from_file['arr_0'], from_file['arr_1']
      print("X shape {}".format(self.X.shape))
      print("y shape{}".format(self.y.shape))

# wrapper class for features model
class Model:
  def __init__(self):
    # construct CNN model
    self.model = models.Sequential()
    self.model.add(layers.Conv2D(16, (1, 1), activation='relu', input_shape=(34, 100, 2)))
    self.model.add(layers.Conv2D(16, (34, 1), activation='relu'))
    self.model.add(layers.Conv2D(32, (1, 100), activation='relu'))

    self.model.add(layers.Flatten())
    self.model.add(layers.Dense(32, activation='relu'))
    self.model.add(layers.Dense(8, activation='relu'))
    self.model.add(layers.Dense(1))

    self.model.summary()

  def train(self, X, y):
    val_idx = random.sample(range(X.shape[0]), 10240)
    train_idx = [idx for idx in range(X.shape[0]) if idx not in val_idx]
    train_dataset = data.Dataset.from_tensor_slices((X[train_idx,:,:,:], y[train_idx]))
    train_dataset = train_dataset.shuffle(300000).batch(128)
    val_dataset = data.Dataset.from_tensor_slices((X[val_idx,:,:,:], y[val_idx]))
    val_dataset = val_dataset.batch(128)

    print("Training Model ...")
    self.model.compile(optimizer = 'adam',
            loss = losses.MeanSquaredError())
    history = self.model.fit(train_dataset, epochs=10, validation_data=val_dataset)

    print("Validating Model ...")
    self.model.evaluate(val_dataset)

    print("Saving Model ...")
    self.model.save('feature_model')

    print("Displaying Loss Plots ...")
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title("Feature Model Loss")
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


if __name__ == "__main__":
  dataset = Dataset('data/feat-dataset') #, 'data/train-features.txt', 'data/train.txt')
  model = Model()
  model.train(dataset.X, dataset.y)
