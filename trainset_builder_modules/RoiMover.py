import numpy as np
import cv2
import logging
from trainset_builder_modules.RoiHolder import RoiHolder
from trainset_builder_modules.MouseDelegate import MouseDelegate

logger = logging.getLogger(__name__)


class MouseRoiMover(object):
    def __init__(self, roi_holder, window_name, mouse_event_manager):
        """
        :type mouse_event_manager: MouseDelegate
        :type roi_holder: RoiHolder
        """
        self.button_pressed = False

        self.roi_holder = roi_holder
        mouse_event_manager.add_to(cv2.EVENT_LBUTTONDOWN, self.react_to_LBUTTONDOWN)
        mouse_event_manager.add_to(cv2.EVENT_LBUTTONUP, self.react_to_LBUTTONUP)
        mouse_event_manager.add_to(cv2.EVENT_MOUSEMOVE, self.react_to_MOUSEMOVE)
        mouse_event_manager.add_to(cv2.EVENT_MOUSEWHEEL, self.react_to_MOUSEWHEEL)

    def react_to_MOUSEWHEEL(self, event, x, y, flags, param):
        self.roi_holder.modifier += float(flags) / (
            float(1258291200))  # really magic number. wheel gives really big numbers

    def react_to_MOUSEMOVE(self, event, x, y, flags, param):
        if self.button_pressed:
            self.roi_holder.center = np.array([x, y])

    def react_to_LBUTTONUP(self, event, x, y, flags, param):
        logger.debug('L button up')
        self.button_pressed = False

    def react_to_LBUTTONDOWN(self, event, x, y, flags, param):
        logger.debug('L button down')
        self.button_pressed = True
        self.roi_holder.center = np.array([x, y])
