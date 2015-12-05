import numpy as np
import cv2

cap = cv2.VideoCapture("C:\\Users\\sht3ch\\Documents\\X-WinFF_1.5.3_rev6\\Documents\\Winff Output\\tmp_8442657.avi")

while(cap.isOpened()):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cv2.imshow('frame',gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()