import numpy as np
import ast

class Dataset:
  def __init__(self, dataset_file, features_file=None, speed_file=None):
    if not (features_file != speed_file):
      raise Exception("You only have one of features_file or speed_file")

    # if features_file is not None, we need to construct the dataset
    if features_file:
      # read in raw feature info into numpy array
      self.X = []
      min_feats = 1000
      with open(features_file) as raw_feats:
        while True:
          num_feats = raw_feats.readline().rstrip()
          if num_feats:
            old_feats = []
            new_feats = []
            for i in range(int(num_feats)):
              # only take the first 200 matching features per frame
              feat_pair = ast.literal_eval(raw_feats.readline().rstrip())
              if i < 50:
                new_feats.append(feat_pair[0] + feat_pair[2])
                old_feats.append(feat_pair[1] + feat_pair[3])
            # new_feats and old_feats form the two "channels" in each datapoint
            self.X.append([new_feats, old_feats])
          # break when we run out of frames      
          else:
            break
        # turn out X into a numy array
        self.X = np.array(self.X, np.uint16)
        print(self.X.shape)

      # read in the speed values into numpy array
      self.y = []
      with open(speed_file) as raw_speed:
        self.y = raw_speed.read().splitlines()
      self.y = [float(line.rstrip()) for line in self.y]
      self.y = np.array(self.y, np.float16)
      print(self.y.shape)

      # save numpy arrays
      np.savez(dataset_file, self.X, self.y)


if __name__ == "__main__":
  dataset = Dataset('data/feat-dataset', 'data/train-features.txt', 'data/train.txt')
