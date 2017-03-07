import os
import time
from glob import glob

import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from vehicle_detection.combine import extract_features
from vehicle_detection.hog_features import get_car_dataset


def train(dataset_name='dataset',
          color_space='YCrCb',  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
          orient=9,  # HOG orientations
          pix_per_cell=8,  # HOG pixels per cell
          cell_per_block=2,  # HOG cells per block
          hog_channel='ALL',  # Can be 0, 1, 2, or "ALL"
          spatial_size=(32, 32),  # Spatial binning dimensions
          hist_bins=32,  # Number of histogram bins
          hog=True,
          spatial=True,
          hist=True,
          dump_filename=None,
          false_positive_path=None
          ):
    # Read in cars and notcars
    data_info = get_car_dataset(dataset_name)  # _small

    dump = dict(
        color_space=color_space,

        hog_channel=hog_channel,
        orient=orient,
        pix_per_cell=pix_per_cell,
        cell_per_block=cell_per_block,

        spatial_size=spatial_size,
        hist_bins=hist_bins,

        # flags to enable the 3 features extraction types
        hog=hog,
        spatial=spatial,
        hist=hist,
    )

    cars = data_info['cars']
    notcars = data_info['notcars']

    # Reduce the sample size because
    # The quiz evaluator times out after 13s of CPU time

    car_features = extract_features(cars,
                                    **dump
                                    )
    notcar_features = extract_features(notcars,
                                       **dump
                                       )
    x_features = [car_features, notcar_features]
    if false_positive_path:
        false_positives = glob(false_positive_path)
        false_features = extract_features(false_positives, **dump)
        x_features.append(false_features)

    X = np.vstack(x_features).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    labels = [np.ones(len(car_features)), np.zeros(len(notcar_features))]
    if false_positive_path:
        labels.append(np.zeros(len(false_features)))
    y = np.hstack(labels)

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
    print('Training SVC...')
    t = time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2 - t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    dump.update(dict(
        svc=svc,
        X_scaler=X_scaler,
    ))

    if dump_filename:
        print("Saving trained classifier and settings to: %s" % dump_filename)
        joblib.dump(dump, dump_filename)
    return dump


def load(trained_dump_filename='train_dump.jot'):
    return joblib.load(trained_dump_filename)


def load_or_train(trained_dump_filename='train_dump.jot'):
    if os.path.isfile(trained_dump_filename):
        dump = load(trained_dump_filename)
    else:
        dump = train(dump_filename='train_dump.jot')
    return dump


def refine(false_positive_path, dump_filename=None):
    dump = load()

    false_positives = glob(false_positive_path)
    svc = dump.pop('svc')
    X_scaler = dump.pop('X_scaler')
    false_features = extract_features(false_positives, **dump)
    X = np.array(false_features).astype(np.float64)
    y = np.zeros(len(false_features))
    print("Refinement training for false positive")
    svc.fit(X, y)
    dump['svc'] = svc
    dump['X_scaler'] = X_scaler
    if dump_filename:
        print("Saving trained classifier and settings to: %s" % dump_filename)
        joblib.dump(dump, dump_filename)


if __name__ == '__main__':
    train(
        dump_filename='train_dump.jot',
        false_positive_path='img/detected_vehicles/false_positive/*'
    )
