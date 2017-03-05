import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.measurements import label
from sklearn.externals import joblib

from combine import find_car_boxes


def get_heatmap(img, bboxes, threshold):
    heatmap = np.zeros_like(img[:, :, 0]).astype(np.float)
    for box in bboxes:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1
    heatmap[heatmap <= threshold] = 0
    return heatmap


def draw_labeled_bboxes(img, labeled_array, num_features):
    out = np.copy(img)
    # Iterate through all detected cars
    for car_number in range(1, num_features + 1):
        # Find pixels with each car_number label value
        nonzero = (labeled_array == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        # Draw the box on the image
        cv2.rectangle(out, bbox[0], bbox[1], (0, 0, 255), 6)
    # Return the image
    return out


if __name__ == '__main__':
    trained_dump_filename = 'train_dump.jot'
    dump = joblib.load(trained_dump_filename)
    print(dump.keys())

    ystart = 400
    ystop = 656
    scale = 1.5
    img = mpimg.imread('img/test_simple.jpg')
    bboxes = find_car_boxes(img,
                            ystart=400, ystop=656,
                            scale=1.5,
                            **dump)

    heatmap = get_heatmap(img, bboxes, 1)
    labeled_array, num_features = label(heatmap)

    print(num_features, 'cars found')
    out = draw_labeled_bboxes(img, labeled_array, num_features)

    fig = plt.figure()
    plt.subplot(121)
    plt.imshow(out)
    plt.title('Car Positions')
    plt.subplot(122)
    plt.imshow(heatmap, cmap='hot')
    plt.title('Heat Map')
    fig.tight_layout()
    plt.show()
