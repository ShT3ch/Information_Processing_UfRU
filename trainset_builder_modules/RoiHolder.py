import numpy as np
import logging
import itertools
import cv2

logger = logging.getLogger(__name__)


class target_polygon(object):
    def __init__(self, points):
        self.points = points

    def get_roi(self):
        minimal_X = self.points[:, 0].min()
        maximal_X = self.points[:, 0].max()
        minimal_Y = self.points[:, 1].min()
        maximal_Y = self.points[:, 1].max()

        logger.debug('get roi from polygon. ' +
                     'minimal_X: [%i]' +
                     'maximal_X: [%i]' +
                     'minimal_Y: [%i]' +
                     'maximal_Y: [%i]',
                     minimal_X, maximal_X, minimal_Y, maximal_Y)

        return np.array(
                [(minimal_X, minimal_Y),
                 (maximal_X, minimal_Y),
                 (maximal_X, maximal_Y),
                 (minimal_X, maximal_Y)])

    def draw_yourself(self, img):
        for point in self.points:
            img = cv2.circle(img, tuple(point), 5, (40, 150, 50), -1)

        prev_point = self.points[0, :]
        for point in itertools.chain(self.points[1:, :], [self.points[0, :]]):
            img = cv2.line(img, tuple(prev_point), tuple(point), (0, 150, 50), 2)
            prev_point = point


class RoiHolder(object):
    def __init__(self):
        self.center = np.array((0, 0))
        self.vector = np.array([0, 0])
        self.length = np.array([0, 0])

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

    def is_corners(self, posible_corners):
        if posible_corners.shape[0] != 2 and posible_corners.shape[0] != 4:
            raise RuntimeError(
                    'corners vector should have 2 or 4 vectors! Actually shape is [%s]' % str(posible_corners.shape))
        return True

    def is_right_order(self, corners):
        def get_angle(p0, p1=np.array([0, 0]), p2=None):
            ''' compute angle (in degrees) for p0p1p2 corner
            Inputs:
                p0,p1,p2 - points in the form of [x,y]
            '''
            if p2 is None:
                p2 = p1 + np.array([1, 0])
            v0 = np.array(p0) - np.array(p1)
            v1 = np.array(p2) - np.array(p1)

            angle = np.math.atan2(np.linalg.det([v0, v1]), np.dot(v0, v1))
            return np.degrees(angle)

        center = corners.mean(0)
        vectors = corners - center
        logger.debug('vectors in right order testing: \t\t[%s]', ', '.join(map(str, vectors)))

        horizontal_direction = np.array([1, 0])

        angles = [get_angle(vec) for vec in vectors]
        logger.debug('corresponding angles with [%s]: \t\t[%s]', horizontal_direction, ', '.join(map(str, angles)))
        logger.debug('corresponding ordered angles with [%s]: \t\t[%s]', horizontal_direction,
                     ', '.join(map(str, sorted(angles))))

        if angles != sorted(angles):
            raise RuntimeError('bad order of corners')

        return True

    def init_vectors_from_diagonal_corners(self, corners=None):
        if corners is None:
            logger.info('diagonal corners not specified, will use internal')
            corners = self.corners

        self.is_corners(corners)
        self.is_right_order(
                corners * (
                    np.array([[1, 0], [0, -1]])))  # [[1, 0], [0, -1]] for fix strange image coordinates with (x, -y)

        logger.info('corners: %s', ', '.join(map(str, corners)))
        self.center = corners.mean(0)
        logger.info('center: %s', self.center)
        vectors = corners - self.center
        logger.info('vectors of frame: %s', ', '.join(map(str, vectors)))
        lengths = np.array([np.linalg.norm(vector) for vector in vectors])
        logger.info('lengths of frame vectors: %s', lengths)

        longest = lengths.max()

        self.length = longest
        self.vector = vectors[0] / lengths[0]

        logger.info('normalized vectors: %s', ', '.join(map(str, self.vector)))

    def get_diagonal_coordinates(self):
        yield self.center - self.vector * self.length * self.modifier
        yield self.center
        yield self.center + self.vector * self.length * self.modifier

    def get_full_coordinates(self):
        for i_th in range(4):
            yield self.center + np.rot90(self.vector, i_th) * self.length
        yield self.center

    def draw_yourself(self, img):
        left_bottom_corner, rect_center, right_upped_corner = self.get_diagonal_coordinates()
        left_bottom_corner = tuple(map(int, left_bottom_corner))
        right_upped_corner = tuple(map(int, right_upped_corner))

        rect_center = tuple(map(int, rect_center))

        with_me = img

        with_me = cv2.rectangle(with_me, left_bottom_corner, right_upped_corner, (100, 200, 0, 0), 4)
        with_me = cv2.circle(with_me, rect_center, 5, (0, 0, 255), -1)

        with_me = cv2.circle(with_me, right_upped_corner, 5, (255, 0, 0), -1)
        with_me = cv2.circle(with_me, left_bottom_corner, 5, (255, 255, 0), -1)

        return with_me, img
