import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import winsound

img = cv2.imread("C:/Users/sht3ch/Pictures/2015-12-11/IMG_4002_.JPG")

rows, cols, ch = img.shape

tracked = [[216, 634], [1746, 181], [1681, 2490], [216, 1914]]

control = [1005, 517]

pts1 = np.float32(tracked)
pts2 = np.float32([[0, 0], [int(1600 / 0.75), 0], [int(1600 / 0.75), 1600], [0, 1600]])

M = cv2.getPerspectiveTransform(pts1, pts2)

dst = cv2.warpPerspective(img, M, (int(1600 / 0.75), 1600))

font = cv2.FONT_HERSHEY_SIMPLEX
for id, point in enumerate(tracked):
    img = cv2.circle(img, tuple(point), 50, (0, 0, 255), -1)
    cv2.putText(img, str(id), tuple(point), font, 4, (255, 255, 255), 2, cv2.LINE_AA)

img = cv2.circle(img, tuple(control), 10, (0, 255, 255), -1)

control.append(1)
vec_contr = np.matrix(control)

print(vec_contr.T)
print(np.matrix(M))

warped_contr = np.matrix(M)*vec_contr.T
print(warped_contr)

coords_of_warped = [int(elem[0]/warped_contr[2,0]) for elem in (warped_contr[:, 0]).tolist()[:2]]

print (coords_of_warped)

dst = cv2.circle(dst, tuple(coords_of_warped), 10, (255, 0, 255), -1)

# cv2.imwrite('C:/Users/sht3ch/Pictures/2015-12-11/corrected.JPG', dst)
plt.subplot(121), plt.imshow(img), plt.title('Input')
plt.subplot(122), plt.imshow(dst), plt.title('Output')
plt.show()
