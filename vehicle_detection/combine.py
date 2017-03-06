import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from .color_features import bin_spatial, color_hist, convert_color
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from vehicle_detection.hog_features import get_car_dataset, get_hog_features


def extract_features(imgs, color_space='RGB',
                     starting_color_space="RGB",
                     spatial_size=(32, 32),
                     hist_bins=32, hist_range=None,
                     orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                     ):
    print("Extracting features:")
    print("Spatial features (resize):", spatial_size)
    print("Color features (hist):", hist_bins, hist_range)
    print("HOG features:", orient, pix_per_cell, cell_per_block, hog_channel)
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    if len(imgs):
        for imgfile in imgs:
            # Read in each one by one
            img = mpimg.imread(imgfile)
            # apply color conversion if other than 'RGB'
            if color_space and color_space != starting_color_space:
                function_name = "COLOR_" + starting_color_space + "2" + color_space
                img = cv2.cvtColor(img, getattr(cv2, function_name))

            # Apply bin_spatial() to get spatial color features
            spatial_features = bin_spatial(img, size=spatial_size)
            # Apply color_hist() to get color histogram features
            hist_features = color_hist(img, nbins=hist_bins, bins_range=hist_range)

            # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
                hog_features = []
                for channel in range(img.shape[2]):
                    hog_features.append(
                        get_hog_features(
                            img[:, :, channel],
                            orient, pix_per_cell, cell_per_block,
                            vis=False, feature_vec=True)
                    )
                hog_features = np.ravel(hog_features)
            else:
                hog_features = get_hog_features(
                    img[:, :, hog_channel], orient,
                    pix_per_cell, cell_per_block, vis=False,
                    feature_vec=True
                )

            # Append the new feature vector to the features list

            features.append(
                np.concatenate(
                    (
                        spatial_features,
                        hist_features,
                        hog_features,
                    )
                )
            )
        print("Feature size {tot} - Spatial: {spatial} - Hist: {hist} - Hog: {hog}".format(
            tot=len(features[0]),
            spatial=len(spatial_features),
            hist=len(hist_features),
            hog=len(hog_features),
        ))
        print()
    # Return list of feature vectors
    return features


def find_car_boxes(img, ystart, ystop, scale,
                   svc, X_scaler,
                   color_space,
                   orient, pix_per_cell, cell_per_block,
                   hog_channel,
                   spatial_size, hist_bins
                   ):
    bboxes = []
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255

    img_tosearch = img[ystart:ystop, :, :]
    ctrans_tosearch = convert_color(img_tosearch, color_space=color_space)
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch,
                                     (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))

    if hog_channel == 'ALL':
        hog_channels = [0, 1, 2]
    else:
        hog_channels = [hog_channel]

    # Define blocks and steps as above
    nxblocks = (ctrans_tosearch.shape[1] // pix_per_cell) - (cell_per_block - 1)
    nyblocks = (ctrans_tosearch.shape[0] // pix_per_cell) - (cell_per_block - 1)
    nfeat_per_block = orient * cell_per_block ** 2
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step

    hogs = []  # a list that will keep the hog features, one per channel
    for chan_number in hog_channels:
        ch = ctrans_tosearch[:, :, chan_number]

        # Compute individual channel HOG features for the entire image
        hog = get_hog_features(ch, orient, pix_per_cell, cell_per_block, feature_vec=False)
        hogs.append(hog)

    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb * cells_per_step
            xpos = xb * cells_per_step
            # Extract HOG for this patch
            hog_feats = []
            for hog in hogs:
                hog_feats.append(
                    hog[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                )
            hog_features = np.hstack(hog_feats)

            xleft = xpos * pix_per_cell
            ytop = ypos * pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop + window, xleft:xleft + window], (64, 64))

            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins, bins_range=None)

            # Scale features and make a prediction
            test_features = X_scaler.transform(
                np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1)
            )
            # test_features = X_scaler.transform(np.hstack((shape_feat, hist_feat)).reshape(1, -1))
            test_prediction = svc.predict(test_features)

            if test_prediction == 1:
                xbox_left = np.int(xleft * scale)
                ytop_draw = np.int(ytop * scale)
                win_draw = np.int(window * scale)
                bboxes.append(
                    ((xbox_left, ytop_draw + ystart),
                     (xbox_left + win_draw, ytop_draw + win_draw + ystart))
                )
    return bboxes


if __name__ == '__main__':
    data_info = get_car_dataset('dataset')  # _small
    cars = data_info['cars']
    notcars = data_info['notcars']

    all = np.concatenate((np.array(cars), np.array(notcars)))
    car_features = extract_features(cars)
    notcar_features = extract_features(notcars)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    car_ind = np.random.randint(0, 100)

    # Define a labels vector based on features lists
    y = np.hstack((np.ones(len(cars)),
                   np.zeros(len(notcars))))

    # dataset ready
    print("dataset_size: {tot} - Cars: {cars} - NotCars: {notcars}".format(
        tot=len(X), cars=len(cars), notcars=len(notcars)
    ))

    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    print("index", car_ind, "=", y[car_ind])

    if False:
        # Plot an example of raw and scaled features
        fig = plt.figure(figsize=(12, 4))
        plt.subplot(131)
        plt.imshow(mpimg.imread(cars[car_ind]))
        plt.title('Original Image')
        plt.subplot(132)
        plt.plot(X[car_ind])
        plt.title('Raw Features')
        plt.subplot(133)
        plt.plot(scaled_X[car_ind])
        plt.title('Normalized Features')
        fig.tight_layout()
        plt.show()

    # Use a linear SVC (support vector classifier)
    svc = LinearSVC()
    # Train the SVC
    svc.fit(X_train, y_train)

    print('Test Accuracy of SVC = ', svc.score(X_test, y_test))
