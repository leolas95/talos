import time
import cv2

__FORMAT_STRING = '%d-%m-%Y--%I:%M:%S%p'


def snapshot(filename, extension='png', *frame):

    # Gets the timestamp, creates the format string and the human readable name
    timestamp = time.time()
    formatted_time = time.strftime(__FORMAT_STRING, time.localtime(timestamp))

    filename = f"{filename}-{formatted_time}.{extension}"

    cv2.imwrite(filename, *frame)
