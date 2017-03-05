import time

import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from combine import extract_features
from hog_features import get_car_dataset


def train(dataset_name='dataset',
          color_space='YCrCb',  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
          orient=9,  # HOG orientations
          pix_per_cell=8,  # HOG pixels per cell
          cell_per_block=2,  # HOG cells per block
          hog_channel=0,  # Can be 0, 1, 2, or "ALL"
          spatial_size=(32, 32),  # Spatial binning dimensions
          hist_bins=32,  # Number of histogram bins
          dump_filename=None,
          ):
    # Read in cars and notcars
    data_info = get_car_dataset(dataset_name)  # _small
    cars = data_info['cars']
    notcars = data_info['notcars']

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time

    car_features = extract_features(cars,
                                    color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel)
    notcar_features = extract_features(notcars,
                                       color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel)
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print('Using:', orient, 'orientations', pix_per_cell,
          'pixels per cell and', cell_per_block, 'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC
    svc = LinearSVC()
    # Check the training time for the SVC
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    if dump_filename:
        dump = dict(
            svc=svc,
            X_scaler=X_scaler,
            orient=orient,
            pix_per_cell=pix_per_cell,
            cell_per_block=cell_per_block,
            spatial_size=spatial_size,
            hist_bins=hist_bins,
            color_space=color_space,
            hog_channel=hog_channel,
        )
        joblib.dump(dump, dump_filename)
    return svc, X_scaler


if __name__ == '__main__':
    train(dump_filename='train_dump.jot')
