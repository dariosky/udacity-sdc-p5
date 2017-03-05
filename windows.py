import cv2
import numpy as np


def draw_boxes(img, bboxes, color=(100, 100, 255), thick=6):
    imcopy = np.copy(img)
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


def slide_window(img, x_start_stop=(None, None), y_start_stop=(None, None),
                 xy_window=(64, 64),
                 xy_overlap=(0.5, 0.5)
                 ):
    # If x and/or y start/stop positions not defined, set to image size
    # Compute the span of the region to be searched
    # Compute the number of pixels per step in x/y
    # Compute the number of windows in x/y
    # Initialize a list to append window positions to
    window_list = []
    overlap_pixels = [int(xy_overlap[0] * xy_window[0]),
                      int(xy_overlap[1] * xy_window[1])]
    xmin, xmax = x_start_stop[0] or 0, (x_start_stop[1] or img.shape[1]) - xy_window[0]
    ymin, ymax = y_start_stop[0] or 0, (y_start_stop[1] or img.shape[0]) - xy_window[1]
    xstep, ystep = xy_window[0] - overlap_pixels[0], xy_window[1] - overlap_pixels[1]
    xmin += ((xmax - xmin) % xstep) // 2  # center in the range
    ymin += ((ymax - ymin) % ystep) // 2
    print(img.shape, xy_window)
    print(ymin, ymax, ystep)
    print(xmin, xmax, xstep)
    for y in range(ymin, ymax + 1, ystep):
        for x in range(xmin, xmax + 1, xstep):
            window_list.append(((x, y), (x + xy_window[0], y + xy_window[1])))

    return window_list
