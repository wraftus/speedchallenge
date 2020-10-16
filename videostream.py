import numpy as np

import cv2
import sdl2
import sdl2.ext

import os

from skimage.measure import ransac
from skimage.transform import FundamentalMatrixTransform

WIDTH = 1920//2
HEIGHT = 1080//2
class Display:
  def __init__(self, W, H):
    sdl2.ext.init()

    self.W, self.H = W, H
    self.window = sdl2.ext.Window("speedchallenge", size=(W,H))
    self.window.show()

  def paint(self, img):
    events = sdl2.ext.get_events()
    for event in events:
      if event.type == sdl2.SDL_QUIT:
        print("Window Closed")
        exit()

    surf = sdl2.ext.pixels3d(self.window.get_surface())
    surf[:, :, 0:3] = img.swapaxes(0,1)

    self.window.refresh()

class Extractor:
  def __init__(self, display):
    self.display = Display(WIDTH, HEIGHT) if display else None
    self.orb = cv2.ORB_create()
    self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

  def extract_features(self, img):
    # use Shi-Tomasi corner detector to find features, amd compute descrip[tors with ORB
    key_points = cv2.goodFeaturesToTrack(np.mean(img, axis=2).astype(np.uint8), 600, qualityLevel=0.01, minDistance=3)
    key_points = [cv2.KeyPoint(kp[0][0], kp[0][1], _size=20) for kp in key_points]
    key_points, desc = self.orb.compute(img, key_points)
    return {'kps': key_points, 'desc': desc}

  def match_features(self, feats1, feats2):
    # use brute force matcher suing Hamming distance
    idx_mapping = []
    matches = self.matcher.knnMatch(feats1['desc'], feats2['desc'], k=2)
    for match1, match2 in matches:
      if match1.distance < 0.75 * match2.distance:
        idx_mapping.append([match1.queryIdx, match1.trainIdx])
    
    # use ransac to get rid of any outlier matches
    if len(idx_mapping) > 0:
      key_points = [[feats1['kps'][idx1].pt, feats2['kps'][idx2].pt] for idx1, idx2 in idx_mapping]
      key_points = np.array(key_points)
      models, inliers = ransac((key_points[:, 0], key_points[:, 1]),
                                FundamentalMatrixTransform,
                                min_samples=8, residual_threshold=1, max_trials=100)
      idx_mapping = np.array(idx_mapping)
      idx_mapping = idx_mapping[inliers]
    return idx_mapping

  def extract_from_video(self, video, out_file):
    # open video file and output file
    cap = cv2.VideoCapture(video)
    with open(out_file, 'w') as output:
      # go through each frame in video
      prev_feats = None
      num_frames = 0
      min_feats = 1000
      while cap.isOpened():
        ret, frame = cap.read()
        if ret:
          # extract features 
          img = cv2.resize(frame, (WIDTH, HEIGHT))
          curr_feats = self.extract_features(img)

          # match features
          if prev_feats:
            idx_mapping = self.match_features(curr_feats, prev_feats)
            print("got {} matches on frame {}".format(len(idx_mapping), num_frames))

            # output features to files
            output.write('{}\n'.format(len(idx_mapping)))
            for idx1, idx2 in idx_mapping:
              x1, y1 = map(lambda x: int(round(x)), curr_feats['kps'][idx1].pt)
              x2, y2 = map(lambda x: int(round(x)), prev_feats['kps'][idx2].pt)
              desc1 = curr_feats['desc'][idx1].tolist()
              desc2 = prev_feats['desc'][idx2].tolist()
              output.write('{}\n'.format([[x1, y1], [x2, y2], desc1, desc2]))

              # add features to be displayed
              if self.display:
                cv2.circle(img, (x1, y1), color=(0, 255, 0), radius=3)
                cv2.line(img, (x1, y1), (x2, y2), color=(255, 0, 0))

            # keep track of minimum features in a fram
            if len(idx_mapping) < min_feats:
                min_feats = len(idx_mapping)

          # draw frame with features to screen
          if self.display:
            self.display.paint(img)
          prev_feats = curr_feats
          num_frames = num_frames + 1

        else:
            break

      print("min features in a frame: {}".format(min_feats))  

if __name__ == "__main__":
  # extract features from video file and save them
  video_file = os.path.join(os.getcwd(), 'data/train.mp4')
  speed_file = os.path.join(os.getcwd(), 'data/train-features.txt')
  extractor = Extractor(False)
  extractor.extract_from_video(video_file, speed_file)
