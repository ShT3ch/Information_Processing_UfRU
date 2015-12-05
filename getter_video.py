import numpy as np
import cv2
import math
import logging
import os

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

root = 'c:\\Users\\sht3ch\\Documents\\viola\\'

relative_pos = 'pos'
path_pos = os.path.join(root, relative_pos)
relative_neg = 'neg'
path_neg = os.path.join(root, relative_neg)
path_annotations = os.path.join(root, 'annos_shtech.ls')
path_bg = os.path.join(root, 'bg_shtech.ls')

cv2.namedWindow('frame')


class CornerHolder(object):
    def __init__(self):
        self.center = np.array((0, 0))
        self.vectors = np.array([0, 0])
        self.length = np.array([0, 0])
        self.modifier = float(1)

        self.corners = np.array([[0, 0], [0, 0]])

        self.current_corner = 0

    def start_corners(self):
        logger.info('start collecting corners')
        self.current_corner = 0
        self.corners = np.array([[0, 0], [0, 0]])
        logger.info('corners: %s', ', '.join(map(str, self.corners)))

    def init_corners(self, coord):
        self.corners[self.current_corner] = coord
        logger.info('corner[%i] %s collected', self.current_corner, coord)
        self.current_corner += 1

        if self.current_corner > 1:
            return False

        return True

    def init_vectors(self):
        logger.info('corners: %s', ', '.join(map(str, self.corners)))
        self.center = self.corners.mean(0)
        logger.info('center: %s', self.center)
        self.vectors = self.corners[0, :] - self.center
        logger.info('result vectors: %s', ', '.join(map(str, self.vectors)))
        self.length = np.linalg.norm(self.vectors)
        logger.info('lengths: %s', self.length)
        self.vectors = np.divide(self.vectors, self.length)
        logger.info('normalized vectors: %s', ', '.join(map(str, self.vectors)))

    def get_coordinates(self):
        yield self.center - self.vectors * self.length * self.modifier
        yield self.center
        yield self.center + self.vectors * self.length * self.modifier


corners = CornerHolder()

corners.start_corners()
corners.init_corners(np.array((400, 400)))
corners.init_corners(np.array((0, 0)))
corners.init_vectors()

global button_pressed
global corners_setting
global space_pressed
global frame_counter
button_pressed = False
corners_setting = False
space_pressed = False
frame_counter = 0


def mouse_click(event, x, y, flags, param):
    global button_pressed
    global corners_setting
    if event == cv2.EVENT_LBUTTONDOWN:
        button_pressed = True
        corners.center = np.array([x, y])
    elif event == cv2.EVENT_LBUTTONUP:
        button_pressed = False
    elif event == cv2.EVENT_MOUSEMOVE:
        if button_pressed:
            corners.center = np.array([x, y])
    elif event == cv2.EVENT_RBUTTONDBLCLK:
        logger.info('double right click. will init corners')
        corners.start_corners()
        corners_setting = True
    elif event == cv2.EVENT_RBUTTONDOWN:
        if corners_setting:
            if not corners.init_corners(np.array([x, y])):
                corners_setting = False
                corners.init_vectors()
    elif event == cv2.EVENT_MOUSEWHEEL:
        corners.modifier += float(flags) / (float(1258291200))


def remember_pos(img, corners):
    """

    :type corners: CornerHolder
    """
    global frame_counter

    left_bottom_corner, _, right_upped_corner = corners.get_coordinates()
    left_bottom_corner = tuple(map(int, left_bottom_corner))
    right_upped_corner = tuple(map(int, right_upped_corner))

    height, width, channels = img.shape

    cropped = img[  max(0, right_upped_corner[1]):min(height, left_bottom_corner[1]),
                    max(0, right_upped_corner[0]):min(width, left_bottom_corner[0])]
    cv2.imshow('cropped', cropped)

    filename = 'pos_%i.jpg' % frame_counter
    cv2.imwrite(os.path.join(path_pos, filename), cropped)
    frame_counter += 1

    filename = os.path.join(relative_pos, filename)
    filemode = 'w'
    if os.path.isfile(path_annotations):
        filemode = 'a'

    logger.info('remembering positive')

    height, width, channels = cropped.shape

    logger.info('left up: [%s]', ' %i %i' % (right_upped_corner[0], right_upped_corner[1]))
    logger.info('right down: [%s]', ' %i %i' % (left_bottom_corner[0], left_bottom_corner[1]))

    with open(path_annotations, filemode) as annotations_file:
        annotations_file.write(filename)
        annotations_file.write(' 1')
        annotations_file.write(' 0 0')
        annotations_file.write(' %i %i' % (width, height))
        annotations_file.write('\n')


def remember_neg(img):
    global frame_counter

    filename = 'neg_%i.jpg' % frame_counter
    cv2.imwrite(os.path.join(path_neg, filename), img)
    frame_counter += 1

    filemode = 'w'
    if os.path.isfile(path_bg):
        filemode = 'a'

    logger.info('remembering background...')

    with open(path_bg, filemode) as bg_file:
        bg_file.write(filename)
        bg_file.write('\n')


cv2.namedWindow('frame')
cv2.setMouseCallback('frame', mouse_click)


def with_frame(img, corner_holder):
    left_bottom_corner, rect_center, right_upped_corner = corner_holder.get_coordinates()
    left_bottom_corner = tuple(map(int, left_bottom_corner))
    right_upped_corner = tuple(map(int, right_upped_corner))
    rect_center = tuple(map(int, rect_center))
    img = cv2.rectangle(img, left_bottom_corner, right_upped_corner, (100, 200, 0, 0), 4)
    img = cv2.circle(img, rect_center, 5, (0, 0, 255), -1)

    img = cv2.circle(img, right_upped_corner, 5, (255, 0, 0), -1)
    img = cv2.circle(img, left_bottom_corner, 5, (255, 255, 0), -1)

    return img


cap = cv2.VideoCapture(0)

while (True):
    global space_pressed
    global button_pressed
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here

    if button_pressed:
        remember_pos(frame, corners)
        # remember_neg(frame)

    frame = with_frame(frame, corners)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == 33:
        space_pressed = True
    else:
        space_pressed = False

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
