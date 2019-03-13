# importing Modules

import cv2
import numpy as np

# Capturing Video through webcam.

cap = cv2.VideoCapture(0)

while(1):
    _, img = cap.read()

    # converting frame(img) from BGR (Blue-Green-Red) to HSV (hue-saturation-value)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # defining the range of Yellow color
    yellow_lower = np.array([40, 40, 10], np.uint8)
    yellow_upper = np.array([100, 255, 255], np.uint8)

    # finding the range yellow colour in the image
    yellow = cv2.inRange(hsv, yellow_lower, yellow_upper)

    # Morphological transformation, Dilation
    kernal = np.ones((5, 5), "uint8")

    blue = cv2.dilate(yellow, kernal)

    res = cv2.bitwise_and(img, img, mask=yellow)

    # Tracking Colour (Yellow)
    (contours, hierarchy) = cv2.findContours(
        yellow, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for pic, contour in enumerate(contours):
        area = cv2.contourArea(contour)
        if(area > 700):

            x, y, w, h = cv2.boundingRect(contour)
            img = cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 3)

    cv2.imshow("Color Tracking", img)
    img = cv2.flip(img, 1)
    cv2.imshow("Yellow", res)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
