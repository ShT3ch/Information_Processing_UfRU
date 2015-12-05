import cv2
import numpy as np
import itertools
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

cv2.namedWindow('frame2', flags=cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
cap = cv2.VideoCapture(0)
#
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

ret, frame1 = cap.read()
previous_frame = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

width, height = (640, 480)

seed = 5
step = 10

points = np.array([np.array(pair) for pair in
                   (itertools.product(list([np.float32(i) for i in range(seed, width, step)]),
                                      list([np.float32(i) for i in range(seed, height, step)])))])

template = np.zeros_like(frame1)

for point in points:
    logger.info('drawing [%s]', point)
    template = cv2.circle(template, (point[0], point[1]), 1, (255, 0, 0), -1)

while True:
    ret, frame2 = cap.read()
    current_frame = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    shifts, statuses, errors = cv2.calcOpticalFlowPyrLK(previous_frame, current_frame, points, None, None, **lk_params)

    flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    for point in points:
        frame2 = cv2.line(frame2, tuple(point), tuple(point + flow[point[1], point[0]]),
                          (0, 30*np.linalg.norm(flow[point[1], point[0]]), 0), 1)

    # mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # out.write(rgb)
    for new_position, status, error, point in zip(shifts, statuses, errors, points):
        if status == 1:
            frame2 = cv2.line(frame2, tuple(point), tuple(np.divide((new_position - point), error) + point),
                              (200, 0, 30 * np.linalg.norm(error)), 1)

    frame2 = cv2.add(frame2, template)

    cv2.imshow('frame2', frame2)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    previous_frame = current_frame

cap.release()
cv2.destroyAllWindows()
