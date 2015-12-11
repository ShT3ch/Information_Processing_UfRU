import logging
import numpy as np
from unittest import TestCase
import itertools
from trainset_builder_modules.RoiHolder import RoiHolder

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class TestRoiHolder(TestCase):
    testing_model = RoiHolder()

    def test_is_right_order(self):
        good_set_4 = [(-1, 1), (1, 1), (1, -1), (-1, -1)]

        good_set_2 = [(-1, 1), (1, -1)]

        sets = [good_set_4, good_set_2]

        for good_set in sets:
            self.assertTrue(self.testing_model.is_right_order(np.array(good_set)))

            for possible in itertools.permutations(good_set):
                corners = list(possible)
                if corners != good_set:
                    self.assertRaises(RuntimeError, self.testing_model.is_right_order, np.array(corners))

    def test_is_corners(self):
        logger.info('testing ROIHolder.is_corner.')
        for i_th in range(1, 10):
            value = np.random.rand(i_th, 2)
            logger.info('testing case [%s][%s] on iteration [%i]', value.shape, u', '.join(map(str,value)), i_th)
            if value.shape[0] == 2 or value.shape[0] == 4:
                self.assertTrue(self.testing_model.is_corners(value))
            else:
                self.assertRaises(RuntimeError, self.testing_model.is_corners, value)

    def test_init_vectors_from_diagonal_corners(self):
        pass

    def test_get_diagonal_coordinates(self):
        pass

    def test_get_full_coordinates(self):
        pass
