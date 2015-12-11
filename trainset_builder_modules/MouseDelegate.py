import cv2
import logging

logger = logging.getLogger(__name__)


class MouseDelegate(object):
    def __init__(self, window_name):
        self.handlers = dict()
        self.window_name = window_name

        cv2.namedWindow(self.window_name)

        cv2.setMouseCallback(self.window_name, self.central_handler)

    def central_handler(self, event, x, y, flags, param):
        logger.debug('mouse event([%s]) on [%s] window', event, self.window_name)
        if event in self.handlers:
            for event_handler in self.handlers[event]:
                event_handler(event, x, y, flags, param)

    def add_to(self, cv_mouse_event_name, handler):
        """
        :param cv_mouse_event_name: cv2.EVENT_LBUTTONDOWN, cv2.EVENT_LBUTTONUP, etc
        :param handler: mouse_handler(event, x, y, flags, param)
        """

        logger.debug('adding handler([%s]) to [%s]', handler, cv_mouse_event_name)

        if cv_mouse_event_name not in self.handlers:
            self.handlers[cv_mouse_event_name] = []
        self.handlers[cv_mouse_event_name] += [handler]

    def remove_from(self, cv_mouse_event_name, handler_to_remove):
        logger.debug('removing handler([%s]) from [%s]', handler_to_remove, cv_mouse_event_name)

        if cv_mouse_event_name not in self.handlers:
            raise RuntimeError(
                "you tried to remove handler[%s] for [%s] window, that was not exist" %
                (cv_mouse_event_name, self.window_name))

        self.handlers[cv_mouse_event_name] = \
            [handler
                for handler in self.handlers[cv_mouse_event_name]
                    if handler != handler_to_remove]
