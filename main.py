# import the necessary packages
import argparse
import time
import sys
import cv2
import imutils
import numpy as np
from imutils.video import FPS, VideoStream
from pyimagesearch.centroidtracker import CentroidTracker
import scipy


def draw_info_on_frame(frame, x1, y1, x2, y2, color, label, labely, p1, p2):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, labely),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.line(frame, p1, p2, (255, 0, 0), 5)
    # text = "ID {}".format(objectID)
    # cv2.putText(
    #     frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    # cv2.circle(
    #     frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)


# Check wether the object in the frame, delimited by (x1, y1), (x2, y2)
# meets all the properties. Returns True if so, False otherwise
def object_meets_criteria(frame, properties, x1, y1, x2, y2, p1, p2):
    properties_fulfilled = []
    from detect_colors import is_red, is_green, is_blue

    for propname, propvalue in properties.items():
        if propname == 'shirt':
            upper_half = frame[y1:p2[1], x1:p2[0]]
            copy = np.copy(upper_half)
        elif propname == 'trousers':
            lower_half = frame[p1[1]:y2, p1[0]:x2]
            copy = np.copy(lower_half)
        else:
            print('ERROR: Property', propname, 'unknown')
            continue

        if propvalue == 'red':
            properties_fulfilled.append(is_red(copy))
        elif propvalue == 'blue':
            properties_fulfilled.append(is_blue(copy))
        elif propvalue == 'green':
            properties_fulfilled.append(is_green(copy))
        else:
            print('ERROR: Property value', propvalue, 'unknown')
            continue

    return False not in properties_fulfilled


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=False,
                help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=False,
                help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

programData = {
    'targets': {
        'chair': {
            'min': 3
        },
        'person': {
            'min': 1,
            'properties': {
                'shirt': 'green',
            }
        }
    }
}

targets = programData['targets'].keys()

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('./MobileNetSSD_deploy.prototxt.txt',
                               './MobileNetSSD_deploy.caffemodel')

# initialize the video stream, allow the cammera sensor to warmup,
# and initialize the FPS counter
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)
fps = FPS().start()

ct = CentroidTracker()
(H, W) = (None, None)

# loop over the frames from the video stream
while True:
    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

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
        if confidence > args["confidence"]:
            # extract the index of the class label from the
            # `detections`, then compute the (x, y)-coordinates of
            # the bounding box for the object
            idx = int(detections[0, 0, i, 1])

            class_name = CLASSES[idx]
            if class_name not in targets:
                continue

            # Increment the amount of objects seen for this class
            class_counter[class_name] = class_counter.get(class_name, 0) + 1
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (x1, y1, x2, y2) = box.astype("int")

            bboxHeight = y2 - y1
            bboxWidth = x2 - x1

            p1 = (x1, int(y1+bboxHeight/2) + 40)
            p2 = (x2, int(y2-bboxHeight/2) + 40)

            label = "{}: {:.2f}%".format(class_name, confidence * 100)
            labely = y1 - 15 if y1 - 15 > 15 else y1 + 15
            # On met condition, should append rect and data
            rects.append(
                [x1, y1, x2, y2, (p1, p2, label, labely, COLORS[idx], class_name)])
            # from detect_colors import *
            # if is_green(copy):
            #     label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            #     labely = y1 - 15 if y1 - 15 > 15 else y1 + 15
            #     rects.append([x1, y1, x2, y2, (p1, p2, label, labely, COLORS[idx])])
            # else:
            #     continue

            # cv2.line(frame, p1, p2, (255, 0, 0), 5)
            # draw the prediction on the frame
            # label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
            # cv2.rectangle(frame, (x1, y1), (x2, y2),
            #               COLORS[idx], 2)
            # y = y1 - 15 if y1 - 15 > 15 else y1 + 15
            # cv2.putText(frame, label, (x1, y),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

            """
            from detect_colors import *
            if is_blue(copy):
                cv2.line(frame, p1, p2, (255, 0, 0), 5)
                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (x1, y1), (x2, y2),
                              COLORS[idx], 2)
                y = y1 - 15 if y1 - 15 > 15 else y1 + 15
                cv2.putText(frame, label, (x1, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)
            """

    # objects = ct.update(rects)
    # print(len(objects))

    # draw both the ID of the object and the centroid of the
    # object on the output frame
    # Only draw the bbox and the info for the objects that met the conditions

    # TODO: Find out a way to display the objects id and centroid
    for (x1, y1, x2, y2, (p1, p2, label, labely, color, targetname)) in rects:
        minimum = programData['targets'][targetname].get('min')
        maximum = programData['targets'][targetname].get('max')
        # Amount of detected objects of a class
        detectedObjects = class_counter[targetname]
        properties = programData['targets'][targetname].get('properties')

        # If there is both minimum and maximum
        if minimum is not None and maximum is not None:
            if detectedObjects >= minimum and detectedObjects <= maximum:
                # If there are properties to look for in the object, we need to
                # know if each of them succedeed or not, so we only draw
                # the bbox if the object meets all the criteria
                if properties is not None:
                    if object_meets_criteria(frame, properties, x1, y1, x2, y2, p1, p2):
                        draw_info_on_frame(frame, x1, y1, x2, y2,
                                           color, label, labely, p1, p2)
                # No properties to look for, so just draw the bbox
                else:
                    draw_info_on_frame(frame, x1, y1, x2, y2,
                                       color, label, labely, p1, p2)
        # Just minimum
        elif minimum is not None and maximum is None:
            if detectedObjects >= minimum:
                if properties is not None:
                    if object_meets_criteria(frame, properties, x1, y1, x2, y2, p1, p2):
                        draw_info_on_frame(frame, x1, y1, x2, y2,
                                           color, label, labely, p1, p2)
                else:
                    draw_info_on_frame(frame, x1, y1, x2, y2,
                                       color, label, labely, p1, p2)
        # Just maximum
        elif minimum is None and maximum is not None:
            if detectedObjects <= maximum:
                if properties is not None:
                    if object_meets_criteria(frame, properties, x1, y1, x2, y2, p1, p2):
                        draw_info_on_frame(frame, x1, y1, x2, y2,
                                           color, label, labely, p1, p2)
                else:
                    draw_info_on_frame(frame, x1, y1, x2, y2,
                                       color, label, labely, p1, p2)
        # No range specified, so just draw the bbox
        else:
            draw_info_on_frame(frame, x1, y1, x2, y2,
                               color, label, labely, p1, p2)

    # if minimum is not None and maximum is not None:
    #     if len(objects) >= minimum and len(objects) <= maximum:
    #         # loop over the tracked objects
    #         for (objectID, centroid) in objects.items():
    #             # draw both the ID of the object and the centroid of the
    #             # object on the output frame
    #             # Only draw the bbox and the info for the objects that met the conditions
    #             for (x1, y1, x2, y2, (p1, p2, label, labely, color)) in rects:
    #                 cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    #                 cv2.putText(frame, label, (x1, labely),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    #                 cv2.line(frame, p1, p2, (255, 0, 0), 5)
    #                 text = "ID {}".format(objectID)
    #                 cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    #                 cv2.circle(
    #                     frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # show the output frame
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
vs.stop()
