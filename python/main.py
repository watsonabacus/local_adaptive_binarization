import sys
import os

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math
from nibalck_thresholding import *




k = 20
blockSize = 11;
threshType = cv2.THRESH_BINARY;
method = BINARIZATION_METHOD.BINARIZATION_WOLF;

def onMouse(event, x, y, flags, userdata ):

#     if( event != CV_EVENT_LBUTTONDOWN )
#             return;
  
    print("x={}, y={}, value={}".format(x, y, userdata[y, x]))
    return
 


def nothing(x):
    pass

filename = sys.argv[-1]
print("filename={}".format(filename))
src = cv2.imread(filename, cv2.IMREAD_GRAYSCALE);
width = src.shape[1];
height = src.shape[0];
# namedWindow("Source", WINDOW_AUTOSIZE);
cv2.namedWindow("Source", cv2.WINDOW_NORMAL);
cv2.resizeWindow("Source", width,height);
cv2.setMouseCallback( "Source", onMouse, src );
cv2.imshow("Source", src);


# namedWindow("Niblack", WINDOW_AUTOSIZE);
cv2.namedWindow("Niblack", cv2.WINDOW_NORMAL);
cv2.resizeWindow("Niblack", width,height);
cv2.setMouseCallback( "Niblack", onMouse, src );
cv2.createTrackbar("k", "Niblack", k, 20, nothing);
cv2.createTrackbar("blockSize", "Niblack", blockSize, 30, nothing);
cv2.createTrackbar("method", "Niblack", method.value, 3, nothing);
cv2.createTrackbar("threshType", "Niblack", threshType, 4, nothing);


while(1):
    
    k = cv2.waitKey(1) & 0xFF
    if k == 27:
        break

    k = cv2.getTrackbarPos('k','Niblack')
    blockSize = cv2.getTrackbarPos('blockSize','Niblack')
    threshType = cv2.getTrackbarPos('threshType','Niblack') #     THRESH_BINARY, THRESH_BINARY_INV,
                                                             #THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV
    method = cv2.getTrackbarPos('method','Niblack')  #BINARIZATION_NIBLACK, BINARIZATION_SAUVOLA, BINARIZATION_WOLF, BINARIZATION_NICK
     
    
    
    k = (k-10)/10;   
    blockSize = blockSize if  blockSize >= 1 else 1         #[-1.0, 1.0]
    blockSize = 2*blockSize + 1; # 3,5,7,...,61
    
    method = BINARIZATION_METHOD(method)

    dst = niBlackThreshold(src, 255, threshType, blockSize, k, method);
    cv2.imshow("Niblack", dst);

cv2.destroyAllWindows()

   
