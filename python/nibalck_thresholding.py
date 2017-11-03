import opecv2
from enum import Enum
import numpy as np

BINARIZATION_METHOD= Enum('BINARIZATION_NIBLACK', 'BINARIZATION_SAUVOLA', 'BINARIZATION_WOLF', 'BINARIZATION_NICK')



def niBlackThreshold( src,  dst,  maxValue,
         type,  blockSize,  k,  binarizationMethod ):

    #Input grayscale image
    assert len(src.shape) == 2
    assert blockSize % 2 == 1 and blockSize > 1
    if (binarizationMethod == BINARIZATION_METHOD.BINARIZATION_SAUVOLA):
        assert src.dtype == np.uint8

#     Compute local threshold (T = mean + k * stddev)
#     using mean and standard deviation in the neighborhood of each pixel
#     (intermediate calculations are done with floating-point precision)
    
#     note that: Var[X] = E[X^2] - E[X]^2
#     Mat mean, sqmean, variance, stddev, sqrtVarianceMeanSum;
#     double srcMin, stddevMax;
    mean = cv2.boxFilter(src, cv2.CV_32F, (blockSize, blockSize),
            (-1,-1), True, cv2.BORDER_REPLICATE);
    sqmean = sqrBoxFilter(src, cv2.CV_32F, (blockSize, blockSize),
            (-1,-1), True, cv2.BORDER_REPLICATE);
    variance = sqmean - mean * mean;
    stddev = np.sqrt(variance);
  
    if binarizationMethod == BINARIZATION_METHOD.BINARIZATION_NIBLACK:
        thresh = mean + stddev * static_cast<float>(k);
    elif binarizationMethod == BINARIZATION_METHOD.BINARIZATION_SAUVOLA:
        thresh = mean.mul(1. + static_cast<float>(k) * (stddev / 128.0 - 1.));
    elif binarizationMethod == BINARIZATION_METHOD.BINARIZATION_WOLF:
        srcMin = src.min()
        stddevMax = stddev.max()
        thresh = mean - k * (mean - srcMin - stddev * (mean - srcMin) / stddevMax);

    elif binarizationMethod ==  BINARIZATION_NICK:
        sqrtVarianceMeanSum = np.sqrt(variance + sqmean)
        thresh = mean + k * sqrtVarianceMeanSum;
    else:
        assert "Unknown binarization method"
   
    thresh.astype(src.dtype)


#     Apply thresholding: ( pixel > threshold ) ? foreground : background
    if type == cv2.THRESH_BINARY or type == cv2.THRESH_BINARY_INV:      # dst = (src > thresh) ? maxval : 0
#         dst = (src > thresh) ? 0 : maxval
        cmpop = cv2.CMP_GT if type == cv2.THRESH_BINARY else cv2.CMP_LE
        mask = cv2.compare(src, thresh, cmpop);
        dst = np.zeros_like(src)
        dst[mask] = maxValue
        break;
    elif type == cv2.THRESH_TRUNC:       # dst = (src > thresh) ? thresh : src
        mask = cv2.compare(src, thresh, cv2.CMP_GT);
        dst = src.copy()
        dst[mask] = thres[mask]
        
    elif type == cv2.THRESH_TOZERO or type == cv2.THRESH_TOZERO_INV:     # dst = (src > thresh) ? src : 0
        cmpop = cv2.CMP_GT if type == THRESH_TOZERO else cv2.CMP_LE
        mask = cv2.compare(src, thresh, cmpop)
        dst = np.zeros_like(src)
        dst[mask] = src[mask]
    else:
        assert "Unknown threshold type"
    return dst
    
