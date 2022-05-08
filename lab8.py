import cv2
import numpy as np


def task_1():
    """taks 1"""
    tom_app = cv2.imread(r'pictures\tomatoes_and_apples.jpg')
    tom_app = cv2.resize(tom_app, (0, 0), fx=0.3, fy=0.3)
    tom_app_hsv = cv2.cvtColor(tom_app, cv2.COLOR_BGR2HSV)

    green_min = (30, 10, 0)
    green_max = (100, 200, 255)
    red_min = (0, 0, 0)
    red_max = (15, 255, 245)

    apples_thresh = cv2.inRange(tom_app_hsv, green_min, green_max)
    tomatoes_thresh = cv2.inRange(tom_app_hsv, red_min, red_max)

    cv2.imshow("original image", tom_app)
    cv2.imshow("apples", apples_thresh)
    cv2.imshow("tomatoes", tomatoes_thresh)
    cv2.waitKey()


def main():
    """sh*t"""
    task_1()


if __name__ == "__main__":
    main()