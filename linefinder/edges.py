import cv2
import numpy as np


def binary(img,
           sobelx_thresh=(20, 100),
           luminosity_thresh=(100, 255),
           saturation_thresh=(100, 255),
           ):
    """ Give a color image, get a binary image with edges combining sobelx gradient
        and tresholded saturation channel and luminosity
    """
    # Convert to HSV color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    l_channel = hls[:, :, 1]
    saturation_channel = hls[:, :, 2]

    # Sobel x over luminosity
    # Absolute x derivative to accentuate lines away from horizontal
    scaled_luminosity = np.uint8(l_channel / np.max(l_channel) * 255)
    sobelx = cv2.Sobel(scaled_luminosity, cv2.CV_64F, 1, 0)
    abs_sobelx = np.absolute(sobelx)
    scaled_sobelx = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

    # Binary composition of the 3 functions with their threshold
    composition = np.zeros_like(sobelx, dtype=np.uint8)
    composition[
        (scaled_sobelx >= sobelx_thresh[0]) & (scaled_sobelx < sobelx_thresh[1])
        |
        (saturation_channel >= saturation_thresh[0]) & (saturation_channel < saturation_thresh[1])
        &
        (scaled_luminosity >= luminosity_thresh[0]) & (scaled_luminosity < luminosity_thresh[1])
        ] = 255
    return composition
