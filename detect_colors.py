import cv2
import numpy as np


def is_blue(frame):
    if frame.size == 0:
        return False
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([100, 80, 70])
    upper_blue = np.array([140, 255, 255])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresholded = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    blue_ratio = int((cv2.countNonZero(mask) / (frame.size/3) * 100))
    return blue_ratio > 10


def is_red(frame):
    if frame.size == 0:
        return False
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresholded = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)
    # Threshold the HSV image to get only blue colors
    # define range of blue color in HSV
    lower_red = np.array([0, 160, 100])
    upper_red = np.array([30, 255, 255])
    mask1 = cv2.inRange(hsv, lower_red, upper_red)

    lower_red = np.array([170, 180, 120])
    upper_red = np.array([180, 255, 255])

    mask2 = cv2.inRange(hsv, lower_red, upper_red)

    mask = mask1 + mask2

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    red_ratio = int((cv2.countNonZero(mask) / (frame.size/3) * 100))
    return red_ratio > 10


def is_green(frame):
    if frame.size == 0:
        return False
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_green = np.array([40, 40, 10])
    upper_green = np.array([100, 255, 255])
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, thresholded = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(frame, frame, mask=mask)
    green_ratio = int((cv2.countNonZero(mask) / (frame.size/3) * 100))
    return green_ratio > 10

# while True:
#     # Take each frame
#     _, frame = cap.read()

#     # Convert BGR to HSV
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

#     # define range of blue color in HSV
#     lower_blue = np.array([100, 80, 70])
#     upper_blue = np.array([140, 255, 255])

#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     ret, thresholded = cv2.threshold(gray, 115, 255, cv2.THRESH_BINARY)

#     # Threshold the HSV image to get only blue colors
#     mask = cv2.inRange(hsv, lower_blue, upper_blue)

#     # Bitwise-AND mask and original image
#     res = cv2.bitwise_and(frame, frame, mask=mask)
#     print(cv2.countNonZero(mask) / (frame.size/3) * 100)

#     cv2.imshow('frame', frame)
#     cv2.imshow('mask', mask)
#     cv2.imshow('res', res)
#     k = cv2.waitKey(5) & 0xFF
#     if k == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
