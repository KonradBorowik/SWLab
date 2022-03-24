import cv2
from matplotlib import pyplot as plt
import numpy as np
import time


def color_value(value):
    print(f"Trackbar reporting for duty with value: {value}")


def todo_1():
    # create a black image, a window
    img = np.zeros((300, 512, 3), dtype=np.uint8)
    cv2.namedWindow('image')

    # create trackbars for color change
    cv2.createTrackbar('R', 'image', 0, 255, color_value)
    cv2.createTrackbar('G', 'image', 0, 255, color_value)
    cv2.createTrackbar('B', 'image', 0, 255, color_value)

    # create switch for ON/OFF functionality
    switch_trackbar_name = '0 : OFF \n1 : ON'
    cv2.createTrackbar(switch_trackbar_name, 'image', 0, 1, color_value)

    while True:
        cv2.imshow('image', img)
        # sleep for 10 ms waiting for user to press some key, return -1 on timeout
        key_code = cv2.waitKey(10)
        if key_code == 27:
            # escape key pressed
            break

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('R', 'image')
        g = cv2.getTrackbarPos('G', 'image')
        b = cv2.getTrackbarPos('B', 'image')
        s = cv2.getTrackbarPos(switch_trackbar_name, 'image')

        if s == 0:
            # assign zeros to all pixels
            img[:] = 0
        else:
            # assign the same BGR color to all pixels
            img[:] = [b, g, r]

    # closes all windows (usually optional as the script ends anyway)
    cv2.destroyAllWindows()


def todo_2():
    cv2.namedWindow("todo 2")
    pic = cv2.imread(r"pictures\kubus.jpg")

    cv2.createTrackbar('threshold', "todo 2", 0, 255, color_value)
    thresh_mode = '0 : to white \n1 : to black'
    cv2.createTrackbar(thresh_mode, "todo 2", 0, 1, color_value)
    modes = [cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV]

    gray_pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)

    while True:
        t = cv2.getTrackbarPos('threshold', "todo 2")
        mode = cv2.getTrackbarPos(thresh_mode, "todo 2")

        threshold = cv2.threshold(gray_pic, t, 255, modes[mode])[1]

        cv2.imshow("todo 2", threshold)

        if cv2.waitKey(100) == ord('q'):
            break


def todo_3():
    cube = cv2.imread(r"pictures\qr.jpg")

    linear_start = time.perf_counter()
    linear = cv2.resize(cube, (0,0), fx=2.75, fy=2.75, interpolation=1)
    linear_stop = time.perf_counter()

    nearest_start = time.perf_counter()
    nearest = cv2.resize(cube, (0,0), fx=2.75, fy=2.75, interpolation=0)
    nearest_stop = time.perf_counter()

    area_start = time.perf_counter()
    area = cv2.resize(cube, (0,0), fx=2.75, fy=2.75, interpolation=3)
    area_stop = time.perf_counter()

    lanczos4_start = time.perf_counter()
    lanczos4 = cv2.resize(cube, (0,0), fx=2.75, fy=2.75, interpolation=4)
    lanczos4_stop = time.perf_counter()

    print(f'linear: {linear_stop - linear_start} \n')
    print(f'nearest: {nearest_stop - nearest_start} \n')
    print(f'area: {area_stop - area_start} \n')
    print(f'lanczos4: {lanczos4_stop - lanczos4_start}')

    cv2.imshow('linear', linear)
    cv2.imshow('nearest', nearest)
    cv2.imshow('area', area)
    cv2.imshow('lanczos4', lanczos4)
    cv2.waitKey(0)


def todo_4():
    cv2.namedWindow('blended')

    mcqueen = cv2.imread(r'pictures\mcqueen.jpg')

    logo = cv2.imread(r'pictures\LOGO_PUT_VISION_LAB_MAIN.png')
    logo = cv2.resize(logo, (1700, 1100), interpolation=cv2.INTER_CUBIC)

    cv2.createTrackbar('alpha', 'blended', 0, 100, color_value)
    cv2.createTrackbar('beta', 'blended', 0, 100, color_value)

    while True:
        alpha = cv2.getTrackbarPos('alpha', 'blended')
        beta = cv2.getTrackbarPos('beta', 'blended')

        blended = cv2.addWeighted(mcqueen, alpha/100, logo, beta/100, 0)

        blended = cv2.resize(blended, (0,0), fx=0.5, fy=0.5, interpolation=1)
        cv2.imshow('blended', blended)

        if cv2.waitKey(100) == ord('q'):
            break

    cv2.destroyAllWindows()


def todo_5():
    mcqueen = cv2.imread(r'pictures\mcqueen.jpg')
    mcqueen = cv2.resize(mcqueen, (0, 0), fx=0.5, fy=0.5)
    negative_mcqueen = cv2.bitwise_not(mcqueen)

    cv2.imshow('negative', negative_mcqueen)
    cv2.waitKey()


def main():
    # todo_1()
    # todo_2()
    # todo_3()
    # todo_4()
    todo_5()


if __name__ == '__main__':
    main()
