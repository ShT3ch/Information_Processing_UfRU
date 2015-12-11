import numpy as np
import cv2
import logging
import os
from trainset_builder_modules.RoiHolder import RoiHolder
from trainset_builder_modules.sample_writers import NegativeExampleRememberer, PositiveExampleRememberer
from trainset_builder_modules.RoiMover import MouseRoiMover
from trainset_builder_modules.MouseDelegate import MouseDelegate

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

mouse_manager = MouseDelegate('frame')
corners = RoiHolder()
root = 'c:\\Users\\sht3ch\\Documents\\viola\\'
neg_ex_rememberer = NegativeExampleRememberer(os.path.join(root, 'neg'), os.path.join(root, 'bg.txt'))
pos_ex_rememberer = PositiveExampleRememberer(os.path.join(root, 'pos'), os.path.join(root, 'annotations.ls'))

global corners_setting
corners_setting = False


def mouse_click(event, x, y, flags, param):
    global corners_setting
    if event == cv2.EVENT_RBUTTONDBLCLK:
        logger.info('double right click. will init corners')
        corners.start_corners()
        corners_setting = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        if corners_setting:
            if not corners.init_corners(np.array([x, y])):
                corners_setting = False
                corners.init_vectors_from_diagonal_corners()


mouse_manager.add_to(cv2.EVENT_RBUTTONDBLCLK, mouse_click)
mouse_manager.add_to(cv2.EVENT_RBUTTONDOWN, mouse_click)
mouse_roi_mover = MouseRoiMover(corners, 'frame', mouse_event_manager=mouse_manager)


def generate_mask(img, corner_holder):
    img2gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
    ans = np.zeros_like(mask)

    left_bottom_corner, _, right_upped_corner = corner_holder.get_diagonal_coordinates()
    left_bottom_corner = tuple(map(int, left_bottom_corner))
    right_upped_corner = tuple(map(int, right_upped_corner))

    height, width = ans.shape

    ans[max(0, right_upped_corner[1]):min(height, left_bottom_corner[1]),
    max(0, right_upped_corner[0]):min(width, left_bottom_corner[0])] = 255

    return ans, img2gray


cap = cv2.VideoCapture(0)
ret, frame = cap.read()

previous_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

while True:
    global space_pressed
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    #
    # if mouse_roi_mover.button_pressed:
    #     pos_ex_rememberer.remember_pos(frame, corners)
    #     # remember_neg(frame)

    roi_mask, current_gray_frame = generate_mask(frame, corners)

    cv2.imshow('mask', roi_mask)

    shift_of_detail = corners.center

    points_to_track = cv2.goodFeaturesToTrack(current_gray_frame, mask=roi_mask, **feature_params)

    if points_to_track is not None:
        for point in points_to_track:
            img = cv2.circle(frame, tuple(point[0]), 5, (0, 120, 200), -1)

        shifts, statuses, errors = cv2.calcOpticalFlowPyrLK(previous_frame, current_gray_frame,
                                                            np.array([point[0] for point in points_to_track]), None,
                                                            None,
                                                            **lk_params)

        shift_of_detail = np.mean(shifts, axis=0)

        corners.center = shift_of_detail

        # logger.info('shift of frame: [%s], shape: [%s], src shape: [%s]', shift_of_detail, shift_of_detail.shape,
        #             shifts.shape)

    with_roi, frame = corners.draw_yourself(frame)

    # Display the resulting frame
    cv2.imshow('frame', with_roi)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('p'):
        logger.info('get tracked image to positive')

    previous_frame = current_gray_frame

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
