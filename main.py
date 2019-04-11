import argparse
import datetime
import time

import cv2
import imutils
import numpy as np
from imutils.video import FPS, VideoStream

import config_file_loader

from activities.detection.detect_activity import detect_activity
from activities.conditions.check import check_activity_conditions

from targets.properties.check import check_target_properties
from targets.tracker.centroidtracker import CentroidTracker
from targets.conditions.check import check_targets_conditions

# construct the argument parse and parse the arguments
ARGUMENT_PARSER = argparse.ArgumentParser()
ARGUMENT_PARSER.add_argument("-c", "--confidence", type=float, default=0.25,
                             help="minimum probability to filter weak detections")

ARGUMENT_PARSER.add_argument("-f", "--file", type=str, default="config.json",
                             help="Name of the configuration file. Defaults to 'config.json' \
                in the directory where the main script is located")

ARGUMENTS = vars(ARGUMENT_PARSER.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
#COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

COLORS = [
    (0, 0, 0),
    (90, 178, 30),  # aeroplane
    (11, 33, 57),  # bicycle
    (23, 42, 13),  # bird
    (12, 99, 44),  # boat
    (55, 98, 90),  # bottle
    (100, 100, 200),  # bus
    (140, 175, 80),  # car
    (150, 100, 140),  # cat
    (250, 0, 140),  # chair
    (100, 90, 134),  # cow
    (96, 78, 45),  # diningtable
    (34, 90, 121),  # dog
    (200, 145, 40),  # horse
    (124, 55, 90),  # motorbike
    (223.39221401, 113.79734267, 17.9394392),  # person
    (60, 88, 44),  # pottedplant
    (99, 170, 115),  # sheep
    (190, 55, 200),  # sofa
    (160, 44, 190),  # train
    (112, 113, 56),  # tvmonitor
]

CURRENT_DATE_FORMAT_STRING = "%A %d %B %Y %I:%M:%S %p"


def main():
    program_data = config_file_loader.load(ARGUMENTS['file'])

    targets = program_data.get('targets')
    if targets is not None:
        targets = targets.keys()

    # Dictionary of counter names specified by the user in the DSL program.
    # Each key is the name of the counter, and the value is a set, whose elements
    # are the IDs of the objects detected
    counters = {}

    targets_conditions = program_data.get('targets_conditions')

    activities = program_data.get('activities')
    if activities is not None:
        activities = activities.keys()

    activities_conditions = program_data.get('activities_conditions')

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(
        './MobileNetSSD_deploy.prototxt.txt', './MobileNetSSD_deploy.caffemodel')

    # initialize the video stream, allow the cammera sensor to warmup,
    # and initialize the FPS counter
    print("[INFO] starting video stream...")
    video_source = VideoStream(src=0).start()
    time.sleep(2.0)
    fps = FPS().start()

    centroid_tracker = CentroidTracker()

    frame1 = video_source.read()
    frame1 = imutils.resize(frame1, width=600)

    frame2 = video_source.read()
    frame2 = imutils.resize(frame2, width=600)

    cv2.namedWindow("Frame")
    cv2.moveWindow("Frame", 380, 150)

    # loop over the frames from the video stream
    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        frame = video_source.read()
        frame = imutils.resize(frame, width=600)

        # Check if there are activities to look for
        if activities is not None:
            activity = detect_activity(frame1, frame2, frame)
            if activity is not None and activity in activities:
                check_activity_conditions(
                    activity, activities_conditions, frame)

        if targets is None:
            current_date = datetime.datetime.now().strftime(CURRENT_DATE_FORMAT_STRING)
            cv2.putText(frame, current_date,
                        (frame.shape[1]-345, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            continue

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(
            frame, (300, 300)), 0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions
        net.setInput(blob)
        detections = net.forward()
        rects = []

        # To keep the count of objects detected for a target class
        class_counter = {}

        # loop over the amount of detected objects
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > ARGUMENTS["confidence"]:
                # extract the index of the class label from the
                # `detections`, then compute the (x, y)-coordinates of
                # the bounding box for the object
                idx = int(detections[0, 0, i, 1])

                class_name = CLASSES[idx]
                if class_name not in targets:
                    continue

                # Increment the amount of objects seen for this class
                class_counter[class_name] = class_counter.get(
                    class_name, 0) + 1

                # Get the bounding box coordinates
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (x1, y1, x2, y2) = box.astype("int")

                bbox_height = y2 - y1

                p1 = (x1, int(y1+bbox_height/2) + 40)
                p2 = (x2, int(y2-bbox_height/2) + 40)

                label = "{}: {:.2f}%".format(class_name, confidence * 100)
                labely = y1 - 15 if y1 - 15 > 15 else y1 + 15
                rects.append(
                    [x1, y1, x2, y2, (p1, p2, label, labely, COLORS[idx], class_name)])

        objects = centroid_tracker.update(rects)

        # For each detected object, draw its bounding box, info, centroid and ID
        # if it meets the specified conditions (if any)
        for (object_id, (centroid, rect)) in objects.items():
            (x1, y1, x2, y2, (p1, p2, label, labely, color, target_name)) = rect

            for target in program_data['targets'][target_name]:
                minimum = target.get('min')
                maximum = target.get('max')
                detected_objects = class_counter.get(target_name)
                properties = target.get('properties')

                # Name of the counter specified by the user in the DSL program
                counter_name = target.get('counter')

                if detected_objects is None:
                    continue

                class_counter[target_name] -= 1

                if targets_conditions is not None:
                    # Check if some condition holds true
                    check_targets_conditions(
                        targets_conditions, counters, frame)

                object_data = {
                    'bounding_box': (x1, y1, x2, y2),
                    'middle_line_coords': (p1, p2),
                    'color': color,
                    'label': label,
                    'labely': labely,
                    'object_id': object_id,
                    'centroid': centroid,
                    'counter_name': counter_name,
                }

                # Specified both minimum and maximum amount of objects
                if minimum is not None and maximum is not None:
                    if minimum <= detected_objects <= maximum:
                        check_target_properties(
                            frame, properties, counters, object_data)

                # Just minimum
                elif minimum is not None and maximum is None:
                    if detected_objects >= minimum:
                        check_target_properties(
                            frame, properties, counters, object_data)

                # Just maximum
                elif minimum is None and maximum is not None:
                    if detected_objects <= maximum:
                        check_target_properties(
                            frame, properties, counters, object_data)

                # Neither minimum nor maximum
                else:
                    # So just check properties onwards
                    check_target_properties(
                        frame, properties, counters, object_data)

        # show the current date on the bottom right corner
        current_date = datetime.datetime.now().strftime(CURRENT_DATE_FORMAT_STRING)
        cv2.putText(frame, current_date,
                    (frame.shape[1]-345, frame.shape[0]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    # do a bit of cleanup
    cv2.destroyAllWindows()
    video_source.stop()


if __name__ == '__main__':
    main()
