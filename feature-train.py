import numpy as np
import ast
import random

from tensorflow.keras import layers, models, losses
from tensorflow import data

import matplotlib.pyplot as plt

# wrapper class to create dataset form features text file
FEATS_PER_EXAMPLE=100
EXAMPLES_PER_FRAME=100
VALIDATION_SIZE=20480
class Dataset:
  def __init__(self, train_file, val_file, features_file=None, speed_file=None):
    if bool(features_file) != bool(speed_file):
      raise Exception("You only have one of features_file or speed_file")

    # if features_file is not None, we need to construct the dataset
    if features_file:
      X = []
      y = []

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
              feat_pair = ast.literal_eval(raw_feats.readline().rstrip())
              new_feats.append(feat_pair[0] + feat_pair[2])
              old_feats.append(feat_pair[1] + feat_pair[3])

            # create examples from feature pairs
            for i in range(EXAMPLES_PER_FRAME):
              rand_idx = random.sample(range(len(new_feats)), FEATS_PER_EXAMPLE)
              shuffled_new_feats = [new_feats[idx] for idx in rand_idx]
              shuffled_old_feats = [old_feats[idx] for idx in rand_idx]
              new_ex = np.array([shuffled_new_feats, shuffled_old_feats], np.uint16)
              X.append(np.swapaxes(new_ex, 0, 2))
              y.append(speeds[num_frame])
            print("\rgot examples from frame {}".format(num_frame := num_frame + 1), end='')    

          # break when we run out of frames      
          else:
            print()
            break

      # slpit X into train and val numpy arrays (# examples, size of feat + 2, # of feats per ex, 2)
      val_idx = random.sample(range(len(X)), VALIDATION_SIZE)
      train_idx = [idx for idx in range(len(X)) if idx not in val_idx]
      self.X_train = np.array([X[idx] for idx in train_idx])
      self.X_val = np.array([X[idx] for idx in val_idx])
      self.y_train = np.array([y[idx] for idx in train_idx])
      self.y_val = np.array([y[idx] for idx in val_idx])

      print("Train X shape {}".format(self.X_train.shape))
      print("Train y shape{}".format(self.y_train.shape))
      print("Validation X shape {}".format(self.X_val.shape))
      print("Validation y shape{}".format(self.y_val.shape))

      # save numpy arrays
      np.savez(train_file, self.X_train, self.y_train)
      np.savez(val_file, self.X_val, self.y_val)

    #else we read the dataset from the file
    else:
      with np.load("{}.npz".format(train_file)) as train_file:
        self.X_train, self.y_train = train_file['arr_0'], train_file['arr_1']
        print("Train X shape {}".format(self.X_train.shape))
        print("Train y shape{}".format(self.y_train.shape))
      
      with np.load("{}.npz".format(val_file)) as val_file:
        self.X_val, self.y_val = val_file['arr_0'], val_file['arr_1']
        print("Validation X shape {}".format(self.X_val.shape))
        print("Validation y shape{}".format(self.y_val.shape))

# wrapper class for features model
class Model:
  def __init__(self):
    # construct CNN model
    self.model = models.Sequential()
    self.model.add(layers.Conv2D(16, (1, 1), activation='relu', input_shape=(34, FEATS_PER_EXAMPLE, 2)))
    self.model.add(layers.Conv2D(32, (34, 1), activation='relu'))
    self.model.add(layers.Conv2D(32, (1, FEATS_PER_EXAMPLE), activation='relu'))

    self.model.add(layers.Flatten())
    self.model.add(layers.Dense(32, activation='relu'))
    self.model.add(layers.Dense(1))

    self.model.summary()

  def train(self, X_train, y_train, X_val, y_val):
    train_dataset = data.Dataset.from_tensor_slices((X_train, y_train))
    X_train, y_train = [], []
    val_dataset = data.Dataset.from_tensor_slices((X_val, y_val))
    X_val, y_val = [], []
    train_dataset = train_dataset.shuffle(500000).batch(128)
    val_dataset = val_dataset.batch(128)

    print("Training Model ...")
    self.model.compile(optimizer = 'adam',
            loss = losses.MeanSquaredError())
    history = self.model.fit(train_dataset, epochs=20, validation_data=val_dataset)

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
  dataset = Dataset('data/train-feats-dataset','data/val-feats-dataset', 'data/train-features.txt', 'data/train.txt')
  model = Model()
  model.train(dataset.X_train, dataset.y_train, dataset.X_val, dataset.y_val)
