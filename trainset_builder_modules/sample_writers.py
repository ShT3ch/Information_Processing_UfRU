import cv2
import os
import logging

logger = logging.getLogger(__name__)


class PositiveExampleRememberer(object):
    def __init__(self, positives_dir, path_to_annotations):
        self.path_to_annotations = path_to_annotations
        self.positive_relative_dir = positives_dir[len(os.path.commonprefix([positives_dir, path_to_annotations])):]

        self.positives_dir = positives_dir
        self.current_frame = 0

        logger.info('Annotation will be saved to [%s]', self.path_to_annotations)
        logger.info('relative path to positive samples: [%s]', self.positive_relative_dir)
        logger.info('path to positive samples: [%s]', self.positives_dir)

    def remember_pos(self, img, corners):
        """
        :type corners: CornerHolder
        """

        cropped, cropped_height, cropped_width = self.crop_image(img, corners)

        cv2.imshow('cropped', cropped)

        filename = 'pos_%i.jpg' % self.current_frame
        cv2.imwrite(os.path.join(self.positives_dir, filename), cropped)

        logger.info('remembering positive')

        filename = os.path.join(self.positive_relative_dir, filename)
        filemode = 'w'
        if os.path.isfile(self.path_to_annotations):
            filemode = 'a'

        with open(self.path_to_annotations, filemode) as annotations_file:
            annotations_file.write(filename)
            annotations_file.write(' 1')
            annotations_file.write(' 0 0')
            annotations_file.write(' %i %i' % (cropped_width, cropped_height))
            annotations_file.write('\n')

        self.current_frame += 1


    def crop_image(self, img, corners):
        left_bottom_corner, _, right_upped_corner = corners.get_coordinates()
        left_bottom_corner = tuple(map(int, left_bottom_corner))
        right_upped_corner = tuple(map(int, right_upped_corner))

        height, width, channels = img.shape
        cropped = img[max(0, right_upped_corner[1]):min(height, left_bottom_corner[1]),
                  max(0, right_upped_corner[0]):min(width, left_bottom_corner[0])]
        cropped_height, cropped_width, channels = cropped.shape
        return cropped, cropped_height, cropped_width

class NegativeExampleRememberer(object):
    def __init__(self, negatives_dir, path_to_list):
        self.negatives_dir = negatives_dir
        self.path_to_list = path_to_list
        self.relative_dir_to_negatives = negatives_dir[len(os.path.commonprefix([negatives_dir, path_to_list])):]

        self.current_frame = 0

        logger.info('Background list will be saved to [%s]', self.path_to_list)
        logger.info('relative path to negatives samples: [%s]', self.relative_dir_to_negatives)
        logger.info('path to negative samples: [%s]', self.negatives_dir)

    def remember_neg(self, img):
        filename = os.path.join(self.relative_dir_to_negatives, u'neg_%i.jpg' % self.current_frame)
        cv2.imwrite(os.path.join(self.negatives_dir, filename), img)

        filemode = 'w'
        if os.path.isfile(self.path_to_list):
            filemode = 'a'

        logger.info('remembering background...')

        with open(self.path_to_list, filemode) as bg_file:
            bg_file.write(filename)
            bg_file.write('\n')

        self.current_frame+=1