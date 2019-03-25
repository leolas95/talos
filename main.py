# import the necessary packages
import argparse
import time

import cv2
import imutils
import numpy as np
from imutils.video import FPS, VideoStream

import actions.actions as actions
import config_file_loader
from handle_properties import handle_properties
from pyimagesearch.centroidtracker import CentroidTracker

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--confidence", type=float, default=0.2,
                help="minimum probability to filter weak detections")

ap.add_argument("-f", "--file", type=str, default="config.json",
                help="Name of the configuration file. Defaults to 'config.json' \
                in the directory where the main script is located")
args = vars(ap.parse_args())

# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Gets the value of the operand, searching in the dict counters.
# The operand must be either an identifier (counter) or an integer


def get_value(operand, counters):
    if type(operand) == int:
        return operand
    # If the operand is an identifier, gets its current value, or default to
    # zero if not present yet
    elif operand.isidentifier():
        # Returns the current value of the counter, or 0 if doesn' exist yet
        result = counters.get(operand)
        return 0 if result is None else len(result)

def main():

    program_data = config_file_loader.load(args['file'])

    targets = program_data['targets'].keys()

    # Dictionary of counter names specified by the user in the DSL program.
    # Each key is the name of the counter, and the value is a set, whose elements
    # are the IDs of the objects detected
    counters = {}

    targets_conditions = program_data['targets_conditions']

    # load our serialized model from disk
    print("[INFO] loading model...")
    net = cv2.dnn.readNetFromCaffe(
        './MobileNetSSD_deploy.prototxt.txt', './MobileNetSSD_deploy.caffemodel')

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

        objects = ct.update(rects)

        # For each detected object, draw its bounding box, info, centroid and ID
        # if it meets the specified conditions (if any)
        for (object_id, (centroid, rect)) in objects.items():
            (x1, y1, x2, y2, (p1, p2, label, labely, color, target_name)) = rect

            minimum = program_data['targets'][target_name].get('min')
            maximum = program_data['targets'][target_name].get('max')
            detected_objects = class_counter.get(target_name)
            properties = program_data['targets'][target_name].get('properties')

            # Name of the counter specified by the user in the DSL program
            counter_name = program_data['targets'][target_name].get('counter')

            if detected_objects is None:
                continue

            class_counter[target_name] -= 1

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

            # List to keep the conditions that weren't true
            remaining_conditions = []

            # Check if some condition holds true
            for condition in targets_conditions:
                # Get value of left operand
                left_operand = condition['condition']['left_operand']
                left_operand_value = get_value(left_operand, counters)

                operator = condition['condition']['operator']

                # Get value of right operand
                right_operand = condition['condition']['right_operand']
                right_operand_value = get_value(right_operand, counters)

                context = {
                    left_operand: left_operand_value,
                    right_operand: right_operand_value
                }

                expression = f"{left_operand} {operator} {right_operand}"
                expression_value = eval(expression, context)

                if expression_value is True:
                    # Execute action
                    action = condition['action']
                    action_args = condition['action_arguments']
                    # If the action requires the frame, append it to the arguments
                    if action in actions.ACTIONS_THAT_REQUIRE_FRAME:
                        action_args.append(frame)

                    actions.do(action, *action_args)
                else:
                    # If the current condition is False, we keep it
                    remaining_conditions.append(condition)

            # Update the list to the conditions that haven't been met
            targets_conditions = remaining_conditions

            # Specified both minimum and maximum amount of objects
            if minimum is not None and maximum is not None:
                if minimum <= detected_objects <= maximum:
                    handle_properties(frame, properties, counters, object_data)

            # Just minimum
            elif minimum is not None and maximum is None:
                if detected_objects >= minimum:
                    handle_properties(frame, properties, counters, object_data)

            # Just maximum
            elif minimum is None and maximum is not None:
                if detected_objects <= maximum:
                    handle_properties(frame, properties, counters, object_data)

            # Neither minimum nor maximum
            else:
                # So just check properties onwards
                handle_properties(frame, properties, counters, object_data)

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
