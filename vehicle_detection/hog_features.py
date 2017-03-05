import glob
import random
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

from skimage.feature import hog
from skimage import color, exposure


# Define a function to return some characteristics of the dataset
def data_look(car_list, notcar_list):
    data_dict = {}
    # Define a key in data_dict "n_cars" and store the number of car images
    data_dict["n_cars"] = len(car_list)
    # Define a key "n_notcars" and store the number of notcar images
    data_dict["n_notcars"] = len(notcar_list)
    # Read in a test image, either car or notcar
    # Define a key "image_shape" and store the test image shape 3-tuple
    randimg = mpimg.imread(random.choice(car_list))
    data_dict["image_shape"] = randimg.shape
    # Define a key "data_type" and store the data type of the test image.
    data_dict["data_type"] = randimg.dtype
    data_dict['cars'] = car_list
    data_dict['notcars'] = notcar_list
    # Return data_dict
    return data_dict


def get_hog_features(img, orient: int = 9, pix_per_cell: int = 8, cell_per_block: int = 2,
                     vis=False, feature_vec=True):
    return hog(img, orientations=orient,
               pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block),
               visualise=vis, feature_vector=feature_vec)


def get_car_dataset(dataset='dataset'):
    images = glob.iglob('%s/**/*' % dataset, recursive=True)
    cars = []
    notcars = []

    for image in images:
        if not Path(image).suffix in ('.png', '.jpeg'):
            continue
        if 'non-vehicles' in image:
            notcars.append(image)
        else:
            cars.append(image)

    data_info = data_look(cars, notcars)
    return data_info


if __name__ == '__main__':
    data_info = get_car_dataset()
    # Just for fun choose random car / not-car indices and plot example images
    car_ind = np.random.randint(0, data_info['n_cars'])
    notcar_ind = np.random.randint(0, data_info['n_cars'])
    car_ind = 4057
    print(car_ind)

    # Read in car / not-car images
    car_image = mpimg.imread(data_info['cars'][car_ind])
    notcar_image = mpimg.imread(data_info['notcars'][notcar_ind])

    if False:
        # Plot the examples
        fig = plt.figure()
        plt.subplot(121)
        plt.imshow(car_image)
        plt.title('Example Car Image')
        plt.subplot(122)
        plt.imshow(notcar_image)
        plt.title('Example Not-car Image')
        plt.show()

    gray = cv2.cvtColor(car_image, cv2.COLOR_RGB2GRAY)
    f, h = get_hog_features(gray,
                            orient=9, pix_per_cell=8, cell_per_block=2, vis=True)
    fig = plt.figure(figsize=(30, 20))
    plt.subplot(131)
    plt.imshow(car_image)
    plt.title('Original')
    plt.subplot(132)
    plt.imshow(gray, 'gray')
    plt.title('Gray')
    plt.subplot(133)
    plt.imshow(h)
    plt.title('HOG')
    plt.show()
