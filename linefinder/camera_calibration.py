import glob
import pickle

import cv2
import numpy as np


def get_calibration_points(images, chessboard_size=(9, 6), display=False, pickle_filename=None):
    try:
        # get the camera calibration from pickled file if available
        # otherwise compute them
        with open(pickle_filename, 'rb') as f:
            objpoints, imgpoints = pickle.load(f)
            return objpoints, imgpoints
    except FileNotFoundError:
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d points in real world space
        imgpoints = []  # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)

                if display:
                    # Draw and display the corners
                    img = cv2.drawChessboardCorners(img, chessboard_size, corners, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(500)
        if display:
            cv2.destroyAllWindows()
        if pickle_filename:
            pickle.dump((objpoints, imgpoints), open(pickle_filename, "wb"))
        return objpoints, imgpoints
