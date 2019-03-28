import os
import subprocess

import cv2


def push_notification(title, description, frame):
    cv2.imwrite('temp.png', frame)
    path = os.path.join(os.getcwd(), 'actions/_push.py')
    subprocess.Popen(["python", path, title, description])
