import numpy as np
import cv2

WALK_MIN = 22
WALK_MAX = 33

RUN_MIN = 35
RUN_MAX = 45


def is_walking(x):
    return WALK_MIN <= x <= WALK_MAX


def is_running(x):
    return RUN_MIN <= x <= RUN_MAX


def dist_map(frame1, frame2):
    """outputs pythagorean distance between two frames"""
    frame1_32 = np.float32(frame1)
    frame2_32 = np.float32(frame2)
    diff32 = frame1_32 - frame2_32

    norm32 = np.sqrt(diff32[:, :, 0]**2 + diff32[:, :, 1] **
                     2 + diff32[:, :, 2]**2)/np.sqrt(255**2 + 255**2 + 255**2)
    dist = np.uint8(norm32*255)
    return dist


def detect_activity(frame1, frame2, frame3):
    dist = dist_map(frame1, frame3)
    frame1[:] = frame2
    frame2[:] = frame3
    # apply Gaussian smoothing
    mod = cv2.GaussianBlur(dist, (9, 9), 5)

    # calculate st dev test
    std_dev = cv2.meanStdDev(mod)[1]

    # cv2.putText(
    #     frame3, f"Standard Deviation - {round(std_dev[0][0], 0)}", (70, 70),
    #     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    if is_walking(std_dev):
        return 'walking'
    elif is_running(std_dev):
        return 'running'
    else:
        return None

def main():
    cap = cv2.VideoCapture(0)
    frame1 = cap.read()[1]
    frame2 = cap.read()[1]

    while True:
        frame3 = cap.read()[1]

        act = detect_activity(frame1, frame2, frame3)
        print(act)

        cv2.imshow('frame', frame3)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
