# frame = cv.imread('./img2.jpg')

# hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
# lower_blue = np.array([100, 80, 70])
# upper_blue = np.array([140, 255, 255])

# gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# ret, thresholded = cv.threshold(gray, 115, 255, cv.THRESH_BINARY)

# # Threshold the HSV image to get only blue colors
# mask = cv.inRange(hsv, lower_blue, upper_blue)

# # Bitwise-AND mask and original image
# res = cv.bitwise_and(frame, frame, mask=mask)
# ratio = cv.countNonZero(mask) / (frame.size/3)
# print('blue pixel percentage:', np.round(ratio*100, 2))