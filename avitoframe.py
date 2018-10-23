import os
import cv2

print(cv2.__version__)
vidcap = cv2.VideoCapture('data/action/walking/person06_walking_d3_uncomp.avi')
success,image = vidcap.read()
count = 0
success = True
os.makedirs('data/action/frames/person06_walking_d3_uncomp')

while success:
  cv2.imwrite("data/action/frames/person06_walking_d3_uncomp/%d.jpg" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print 'Read a new frame: ', success
  count += 1