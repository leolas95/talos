# import the necessary packages
import argparse
import time

import imutils
from imutils.video import FPS, VideoStream

import cv2
import numpy as np

from handlers import handle_properties
from pyimagesearch.centroidtracker import CentroidTracker

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
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

program_data = {
    'targets': {
        'chair': {
            'min': 1,
            'max': 3,
            'counter': 'chair-counter'
        },
        'person': {
            'counter': 'person-counter'
        }
    }
}

counters = {}

targets = program_data['targets'].keys()

def main():
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

        # print('rectangulos:', len(rects))
        objects = ct.update(rects)

        # draw both the ID of the object and the centroid of the
        # object on the output frame
        # Only draw the bbox and the info for the objects that met the conditions

        for (index, (object_id, (centroid, rect))) in enumerate(objects.items()):
            (x1, y1, x2, y2, (p1, p2, label, labely, color, targetname)) = rect

            minimum = program_data['targets'][targetname].get('min')
            maximum = program_data['targets'][targetname].get('max')
            detected_objects = class_counter.get(targetname)
            properties = program_data['targets'][targetname].get('properties')

            counter_name = program_data['targets'][targetname].get('counter')

            if detected_objects is None:
                continue

            class_counter[targetname] -= 1
            if minimum is not None and maximum is not None:
                if minimum <= detected_objects <= maximum:
                    handle_properties(frame, properties, (x1, y1, x2, y2), p1, p2,
                                    color, label, labely, object_id, centroid, counter_name, counters)
            elif minimum is not None and maximum is None:
                if detected_objects >= minimum:
                    handle_properties(frame, properties, (x1, y1, x2, y2), p1, p2,
                                    color, label, labely, object_id, centroid, counter_name, counters)
            elif minimum is None and maximum is not None:
                if detected_objects <= maximum:
                    handle_properties(frame, properties, (x1, y1, x2, y2), p1, p2,
                                    color, label, labely, object_id, centroid, counter_name, counters)
            else:
                # Check properties onwards
                handle_properties(frame, properties, (x1, y1, x2, y2), p1, p2,
                                color, label, labely, object_id, centroid, counter_name, counters)

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
    print(counters)

if __name__ == '__main__':
    main()
