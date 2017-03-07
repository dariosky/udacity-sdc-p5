import glob
import os
import statistics

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from blinker import signal
from matplotlib.axes import Subplot
from moviepy.editor import VideoFileClip
from scipy.ndimage.measurements import label

from linefinder.birdeye import get_dstbox, asphalt_box
from linefinder.camera_calibration import get_calibration_points
from linefinder.lines import get_line_points, radius_in_meters, line_on_the_road
from linefinder.edges import binary
from vehicle_detection.combine import find_car_boxes
from vehicle_detection.heat import get_heatmap, draw_labeled_bboxes
from vehicle_detection.train import load_or_train

plt.rcParams["figure.figsize"] = (20, 10)


class Steps:
    original = -1
    undistort = 0
    warp = 1
    binary = 2
    detect_lines = 3
    lines_on_road = 4
    lanes_and_other_cars = 5


ALL_STEPS = (
    Steps.original,
    Steps.undistort,
    Steps.warp,
    Steps.binary,
    Steps.detect_lines,
    Steps.lines_on_road,
    Steps.lanes_and_other_cars,
)


class ProcessPipeline:
    def __init__(self) -> None:
        self.camera = None  # the Camera description
        self.M = None  # distortion matrix for the birdeye view
        self.Minv = None  # inverted distortion matrix

        self.output_prefix = ""
        self.save_steps = False
        self._save_step = 1

        self._plot_step = 0  # counter for the plot
        self.plot_axes = None

        self.YM_PER_PIX = 30 / 720  # meters per pixel in y dimension
        self.XM_PER_PIX = 3.7 / 700  # meters per pixel in x dimension
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.message = self.tracking_message = ""
        self.message_subscribe()

        self.previous_centers = None
        self.previous_fit = None

        self.radius = 0

        # Vehicle detection: classificator and memory
        self.classificator = load_or_train()
        self.detection_params = dict(
            ystart=400,
            ystop=680,
            scale=1.5,
            **self.classificator
        )
        self.rolling_heat = None
        self.overlay_heat = False # True to display the heat overlay
        self.mask_of_interest = None

    def message_subscribe(self):
        """ Subscribe to lane_message events, they can arrive from the pipeline """
        signal('lane_message').connect(self.on_lane_message)
        signal('track_message').connect(self.on_track_message)

    def on_lane_message(self, sender, message=""):
        # print("Message:", message)
        self.message = message

    def on_track_message(self, message=""):
        # print("Tracking:", message)
        self.tracking_message = message

    def camera_calibration(self, img_size):
        images = glob.glob('camera_cal/calibration*.jpg')  # Make a list of calibration images
        objpoints, imgpoints = get_calibration_points(images,
                                                      chessboard_size=(9, 6),
                                                      display=False,
                                                      pickle_filename='calibration_points.pickle')
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                           img_size,
                                                           None, None)
        self.camera = dict(
            mtx=mtx,
            dist=dist
        )

    def save_camera_calibration_examples(self, example_path='camera_cal/calibration1.jpg'):
        img = mpimg.imread(example_path)
        self.img_size = img.shape[:2]
        self.ensure_calibrated()
        print("Saving camera undistortion example")
        mpimg.imsave('output_images/camera_1_original.jpg', img)
        mtx, dist = self.camera['mtx'], self.camera['dist']
        out = cv2.undistort(img, mtx, dist, None, mtx)
        mpimg.imsave('output_images/camera_2_undistorted.jpg', out)

    def ensure_calibrated(self):
        if self.camera is None:
            self.camera_calibration(self.img_size)

    def load(self, img):
        if img.dtype != 'uint8':
            img = (img * 255).astype('uint8')
        if len(img.shape) == 3 and img.shape[-1] == 4:
            img = img[:, :, :3]  # discard alpha channel
        self.img = img  # keep the original image with the process
        self.img_size = self.img.shape[:2]
        self.mask_of_interest = self.get_mask()
        return img

    def save(self, img, name):
        step = self._save_step
        out_file_template = "output_images/{prefix}_step{step}_{name}.jpg"
        filename = out_file_template.format(step=step,
                                            name=name,
                                            prefix=self.output_prefix
                                            )
        print("Saving step: %s" % filename)
        mpimg.imsave(filename, img)
        self._save_step += 1

    def plot(self, img):
        step = self._plot_step
        axe = self.plot_axes
        if not isinstance(axe, Subplot):
            axe = self.plot_axes[step]
        axe.imshow(img)
        self._plot_step += 1

    def undistort(self, img):
        mtx, dist = self.camera['mtx'], self.camera['dist']
        return cv2.undistort(img, mtx, dist, None, mtx)

    def process(self, img,
                plot_steps=(),
                ):
        img = self.load(img)
        self.message = ""  # clean message
        self.tracking_message = ""

        if plot_steps:
            f, self.plot_axes = plt.subplots(len(plot_steps))
            f.subplots_adjust(hspace=0)

        if self.save_steps:
            self.save(img, 'original')
        if Steps.original in plot_steps:
            self.plot(img)

        self.ensure_calibrated()
        self.undist = self.undistort(img)

        if self.save_steps:
            self.save(self.undist, 'undistorted')
        if Steps.undistort in plot_steps:
            self.plot(self.undist)

        if self.M is None:
            dstbox = get_dstbox(asphalt_box, shape=self.img_size)
            srcbox_xy = np.array([(x, y) for y, x in asphalt_box])
            dstbox_xy = np.array([(x, y) for y, x in dstbox])

            # let's get the 2 matrix to get the birdeye and back
            self.M = cv2.getPerspectiveTransform(srcbox_xy, dstbox_xy)
            self.Minv = cv2.getPerspectiveTransform(dstbox_xy,
                                                    srcbox_xy)

        # if Steps.undistort in plot_steps:
        #     # when we plot the warp, let's put the ticks in the undist
        #     for x, y in asphalt_box:
        #         plt.plot(
        #             y, x, 'o', color='red', markersize=6
        #         )

        self.warped = cv2.warpPerspective(self.undist, self.M,
                                          self.img_size[::-1],
                                          flags=cv2.INTER_LINEAR)
        if self.save_steps:
            self.save(self.warped, 'birdeye')
        if Steps.warp in plot_steps:
            self.plot(self.warped)

        self.binary = binary(self.warped)

        if self.save_steps:
            self.save(self.binary, 'binary')
        if Steps.binary in plot_steps:
            self.plot(self.binary)

        # get the lines points
        y, leftx, rightx, extra = get_line_points(
            self.binary,
            self.previous_centers,
            previous_fit=self.previous_fit,
        )

        # save the previous starting point
        self.previous_centers = extra['left_pos'], extra['right_pos']
        self.previous_fit = extra['left_fit'], extra['right_fit']

        if not all([extra['left_pos'], extra['right_pos']]):
            signal('lane_message').send("Can't find lane")
            self.radius = None
            self.distance_from_center = None
        else:
            lane_center = statistics.mean([extra['left_pos'], extra['right_pos']])
            self.distance_from_center = (self.img_size[1] // 2 - lane_center) * self.XM_PER_PIX
            # get the curvature radius (in meters) - average of left/right
            self.radius = statistics.mean(
                radius_in_meters(y, leftx, rightx)
            )

        self.gray = cv2.cvtColor(self.warped, cv2.COLOR_RGB2GRAY)

        if self.save_steps or Steps.detect_lines in plot_steps:
            warped_lines = line_on_the_road(self.gray,
                                            self.undist,
                                            self.Minv,
                                            y, leftx, rightx,
                                            unwarp=False)
            if self.save_steps:
                self.save(warped_lines, 'warped_detection')

            if Steps.detect_lines in plot_steps:
                self.plot(warped_lines)

        out = line_on_the_road(self.gray,
                               self.undist,
                               self.Minv,
                               y, leftx, rightx,
                               )

        if self.radius:
            cv2.putText(out, "radius {:.0f}m".format(self.radius), (50, 50),
                        self.font, 1, (255, 100, 100), 2, cv2.LINE_AA)
        if self.distance_from_center:
            cv2.putText(out, "shift {:.2f}m".format(self.distance_from_center), (50, 85),
                        self.font, 1, (255, 100, 100), 2, cv2.LINE_AA)
        cv2.putText(out, self.message, (self.img_size[1] // 2, 65),
                    self.font, 1.2, (255, 100, 100), 2, cv2.LINE_AA)

        if self.save_steps:
            self.save(out, 'detection')
        if Steps.lines_on_road in plot_steps:
            self.plot(out)

        # P5. Vehicles detection and tracking
        out = self.detect_and_heat(self.undist, out)
        cv2.putText(out, self.tracking_message,
                    (50, 120),
                    self.font, 1.2, (255, 100, 100), 2, cv2.LINE_AA)
        if Steps.lanes_and_other_cars in plot_steps:
            self.plot(out)

        if plot_steps:
            plt.show()
        return out

    def area_of_interest(self):
        height, width = self.img.shape[:2]
        # return a trapezoind on bottom image to around the center of image

        # [(0, 540), (384.0, 270.0), (576.0, 270.0), (960, 540)]
        vertices = [(0, height),
                    (0, height * 0.6),
                    (int(width * 0.35), height * 0.55),
                    (int(width * 0.65), height * 0.55),
                    (width, height * 0.6),
                    (width, height)]
        return np.array([vertices],
                        dtype=np.int32)

    def get_mask(self):
        # defining a blank mask to start with
        mask = np.zeros(self.img.shape[:2], dtype=np.float)
        vertices = self.area_of_interest()
        cv2.fillPoly(mask, vertices, 1)
        return mask

    def detect_and_heat(self, src, out):
        SINGLE_FRAME_THRESHOLD = 1
        ROLLING_HEAT_THRESHOLD = 5
        MAXIMUM_HEAT = 15
        HEAT_GENERATION_DECAY = 1

        if self.rolling_heat is None:
            self.rolling_heat = np.zeros_like(self.gray, dtype=np.float)
        else:
            self.rolling_heat -= HEAT_GENERATION_DECAY  # cool down the previous heat
            self.rolling_heat = np.clip(self.rolling_heat, 0, MAXIMUM_HEAT)

        bboxes = find_car_boxes(src, **self.detection_params)

        heatmap = get_heatmap(self.undist, bboxes, threshold=SINGLE_FRAME_THRESHOLD)
        # heatmap *= self.mask_of_interest
        self.rolling_heat += heatmap  # add the current heatmap

        thresholded_rolling_heat = self.rolling_heat.copy()

        thresholded_rolling_heat[thresholded_rolling_heat < ROLLING_HEAT_THRESHOLD] = 0

        # threshold over the global heat
        labeled_array, num_features = label(thresholded_rolling_heat)

        # mask = np.dstack(
        #     (self.mask_of_interest, self.mask_of_interest, self.mask_of_interest * 255)
        # ).astype(np.uint8)

        # some additional optional overlay - the heatmap and the mask
        # out = cv2.addWeighted(out, 1, mask, .1, 0)

        # HEAT overlay
        if self.overlay_heat:
            out_heat = np.dstack(
                (self.rolling_heat * 200, self.rolling_heat, self.rolling_heat)
            ).astype(np.uint8)
            out = cv2.addWeighted(out, 1, out_heat, .4, 0)
        # ***

        out = draw_labeled_bboxes(out, labeled_array, num_features)
        signal('track_message').send(
            'cars: {tot_cars}'.format(
                tot_cars=num_features,
                # minheat=self.rolling_heat.min(),
                # maxheat=self.rolling_heat.max()
            )
        )

        if self.save_steps:
            self.save(out, 'vehicle_tracking')
        return out


def run_single(filename, show_steps=True, save_steps=False):
    img = mpimg.imread(filename)
    pipeline = ProcessPipeline()

    if save_steps:  # set to True to enable extra output_images saves
        pipeline.save_steps = True
        pipeline.save_camera_calibration_examples()

    pipeline.output_prefix = os.path.splitext(os.path.basename(filename))[0]

    pipeline.process(
        img,
        plot_steps=(
            # Steps.original,
            # Steps.undistort,
            # Steps.warp,
            # Steps.binary,
            # Steps.detect_lines,
            # Steps.lines_on_road,
            Steps.lanes_and_other_cars,
        ) if show_steps else ()
    )


def run_video(filename='video/project_video.mp4'):
    pipeline = ProcessPipeline()
    pipeline.output_prefix = os.path.splitext(os.path.basename(filename))[0]

    def process_image(image):
        out = pipeline.process(image, plot_steps=[])
        return out

    clip1 = VideoFileClip(filename)  # .subclip(t_end=1)
    white_clip = clip1.fl_image(process_image)
    output_video_filename = 'video/{prefix}_output.mp4'.format(prefix=pipeline.output_prefix)
    white_clip.write_videofile(output_video_filename, audio=False)


def run_sequence():
    image_sequence = sorted(glob.glob("img/sequence/*.jpg"))
    pipeline = ProcessPipeline()
    pipeline.output_prefix = "sequence"
    for filename in image_sequence:
        # run_single(filename)
        img = mpimg.imread(filename)
        print(filename)
        out = pipeline.process(img)
        plt.imshow(out)
        plt.show()


def extract_sequence(filename='video/project_video.mp4'):
    clip = VideoFileClip(filename).subclip(t_end=5)  # get the start of video

    for t in range(10):
        filename = "img/sequence/frame_%0d.jpg" % t
        print("Extracting frame", filename)
        clip.save_frame(filename, t=t / 2)  # save every half second


if __name__ == '__main__':
    # to collect the steps and grab a sequence of frame use the following commands
    # run_single('img/test1.jpg', save_steps=True)
    # extract_sequence()

    # examples on single frames (it will do a "blind scan")
    # run_single('img/001240.png')
    # run_single('img/test6.jpg')
    # run_single('img/test5.jpg')
    # ... and an hard one
    # run_single('img/sequence/frame_0.jpg')

    # examples on project video
    run_video('video/test_video.mp4')
    # run_video('video/project_video.mp4')
    # run_video('video/challenge_video.mp4')

    # run_sequence()
