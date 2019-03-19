import cv2
import numpy as np


def draw_info_on_frame(frame, object_data):
    (x1, y1, x2, y2) = object_data['bounding_box']
    color = object_data['color']
    label = object_data['label']
    labely = object_data['labely']
    # (p1, p2) = object_data['middle_line_coords']
    object_id = object_data['object_id']
    centroid = object_data['centroid']

    # Draw the bounding box, target name and confidence
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame, label, (x1, labely),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw the object id and centroid
    text = "ID {}".format(object_id)
    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
    # cv2.line(frame, p1, p2, (255, 0, 0), 5)


# Checks wether the object in the frame, delimited by (x1, y1), (x2, y2)
# meets all the properties. Returns True if so, False otherwise
def object_meets_criteria(frame, properties, bbox, middle_line_coords):
    (x1, y1, x2, y2) = bbox
    (p1, p2) = middle_line_coords
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


def handle_counter(counters, counter_name, object_id):
    # Checks if the counter_name key already exists, and if not, initialize it
    exists = counters.get(counter_name, False)
    counters[counter_name] = set() if not exists else counters[counter_name]
    counters[counter_name].add(object_id)


def handle_properties(frame, properties, counters, object_data):
    object_id = object_data['object_id']
    counter_name = object_data['counter_name']

    # If the object meets all the specified properties, draw the bbox
    if properties is not None:
        if object_meets_criteria(frame, properties, object_data['bounding_box'],
                                 object_data['middle_line_coords']):
            draw_info_on_frame(frame, object_data)

            # If there is a counter specified for this class, update its value
            if counter_name is not None:
                handle_counter(counters, counter_name, object_id)
    # Otherwise, there wasn't specific properties, so just draw the bbox
    else:
        draw_info_on_frame(frame, object_data)
        if counter_name is not None:
            handle_counter(counters, counter_name, object_id)
