import sys
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from nibalck_thresholding import niBlackThreshold

img_file = '/home/levin/workspace/snrprj/snr/data/process_result/snrimgs/region/20171017172348_F017F16221_top_right.bmp'

img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)

img_thres = niBlackThreshold(img)
plt.subplot(1,2,1)
plt.imshow(img, 'gray')
plt.subplot(1,2,2)
plt.imshow(img_thres, 'gray')
plt.show()