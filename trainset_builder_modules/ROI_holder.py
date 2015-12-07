import numpy as np
import logging
import cv2

logger = logging.getLogger(__name__)


class RoiHolder(object):
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

    def draw_yourself(self, img):
        left_bottom_corner, rect_center, right_upped_corner = self.get_coordinates()
        left_bottom_corner = tuple(map(int, left_bottom_corner))
        right_upped_corner = tuple(map(int, right_upped_corner))

        rect_center = tuple(map(int, rect_center))

        with_me = img

        with_me = cv2.rectangle(with_me, left_bottom_corner, right_upped_corner, (100, 200, 0, 0), 4)
        with_me = cv2.circle(with_me, rect_center, 5, (0, 0, 255), -1)

        with_me = cv2.circle(with_me, right_upped_corner, 5, (255, 0, 0), -1)
        with_me = cv2.circle(with_me, left_bottom_corner, 5, (255, 255, 0), -1)

        return with_me, img
