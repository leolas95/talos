import cv2
import time


def snapshot(filename, frame):
    name, extension = filename.split('.')

    # Gets the timestamp, creates the format string and the human readable name
    timestamp = time.time()
    formatstr = '%d-%m-%Y--%I:%M:%S%p'
    formatted_time = time.strftime(formatstr, time.localtime(timestamp))
    filename = f"{name}-{formatted_time}.{extension}"

    cv2.imwrite(filename, frame)
