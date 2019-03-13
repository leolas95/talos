import numpy as np
import time
import cv2

cap = cv2.VideoCapture(0)

time.sleep(3)

boundaries = [
    ([17, 15, 100], [50, 56, 200]),
    ([86, 31, 4], [220, 88, 50]),
    ([25, 146, 190], [62, 174, 250]),
    ([103, 86, 65], [145, 133, 128])
]

ret, frame = cap.read()
for (lower, upper) in boundaries:
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    mask = cv2.inRange(frame, lower, upper)
    out = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.imshow('asd', np.hstack([frame, out]))
    key = cv2.waitKey(0)


cap.release()
cv2.destroyAllWindows()
