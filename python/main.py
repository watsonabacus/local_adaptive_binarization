import sys
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

img_file = './data/20171017162707_F048F13520_top_right.bmp'

img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

# img = cv2.resize(img, (int(img.shape[1]/3), int(img.shape[0]/3)))
plt.imshow(img[...,::-1])
plt.show()