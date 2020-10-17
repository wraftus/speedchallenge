import numpy as np
import ast

from tensorflow.keras import layers, models, losses
from tensorflow import data

# wrapper class to create dataset form features text file
class Dataset:
  def __init__(self, dataset_file, features_file=None, speed_file=None):
    if bool(features_file) != bool(speed_file):
      raise Exception("You only have one of features_file or speed_file")

    # if features_file is not None, we need to construct the dataset
    if features_file:
      # read in raw feature info into numpy array
      self.X = []
      min_feats = 1000
      with open(features_file) as raw_feats:
        num_frame = 0
        while True:
          num_feats = raw_feats.readline().rstrip()
          if num_feats:
            old_feats = []
            new_feats = []
            for i in range(int(num_feats)):
              # only take the first 200 matching features per frame
              feat_pair = ast.literal_eval(raw_feats.readline().rstrip())
              if i < 100:
                new_feats.append(feat_pair[0] + feat_pair[2])
                old_feats.append(feat_pair[1] + feat_pair[3])
            # new_feats and old_feats form the two "channels" in each datapoint
            self.X.append([new_feats, old_feats])
            print("converted frame {}".format(num_frame := num_frame + 1))
          # break when we run out of frames      
          else:
            break
        # turn out X into a numpy array (# frames, # of feats, size of feat + 2, 2)
        self.X = np.array(self.X, np.uint16)
        self.X = np.swapaxes(self.X, 1, 3)
        print(self.X.shape)

      # read in the speed values into numpy array
      self.y = []
      with open(speed_file) as raw_speed:
        self.y = raw_speed.read().splitlines()
      self.y = [float(line.rstrip()) for line in self.y[1:]]
      self.y = np.array(self.y, np.float16)
      print(self.y.shape)

      # save numpy arrays
      np.savez(dataset_file, self.X, self.y)

    #else we read the dataset from the file
    else:
      from_file = np.load("{}.npz".format(dataset_file))
      self.X, self.y = from_file['arr_0'], from_file['arr_1']
      print(self.X.shape)
      print(self.y.shape)

# wrapper class for features model
class Model:
  def __init__(self):
    # construct CNN model
    self.model = models.Sequential()
    self.model.add(layers.Conv2D(32, (1, 1), activation='relu', input_shape=(34, 100, 2)))
    
    self.model.add(layers.Conv2D(32, (2, 1), strides=(2, 1), activation='relu')) 
    self.model.add(layers.Conv2D(64, (2, 1), strides=(2, 1), activation='relu')) 
    self.model.add(layers.Conv2D(64, (2, 1), strides=(2, 1), activation='relu')) 
    self.model.add(layers.Conv2D(128, (2, 1), strides=(2, 1), activation='relu')) 
    self.model.add(layers.Conv2D(128, (2, 1), strides=(2, 1), activation='relu')) 

    self.model.add(layers.Conv2D(64, (1, 1), activation='relu'))
    self.model.add(layers.Conv2D(32, (1, 1), activation='relu'))
    self.model.add(layers.Conv2D(16, (1, 1), activation='relu'))
    self.model.add(layers.Conv2D(8, (1, 1), activation='relu'))
    self.model.add(layers.Conv2D(1, (1, 1), activation='relu'))

    self.model.add(layers.Flatten())
    self.model.add(layers.Dense(50, activation='relu'))
    self.model.add(layers.Dense(25, activation='relu'))
    self.model.add(layers.Dense(1, activation='relu'))

    self.model.summary()

  def train(self, X, y):
    dataset = data.Dataset.from_tensor_slices((X,y))
    dataset = dataset.shuffle(10000).batch(256)

    self.model.compile(optimizer = 'adam',
            loss = losses.MeanSquaredError(),
            metrics = ['accuracy'])
    self.model.fit(dataset, epochs=20)

    print("Saving Model")
    self.model.save('feature_model')

if __name__ == "__main__":
  dataset = Dataset('data/feat-dataset')
  model = Model()
  model.train(dataset.X, dataset.y)
