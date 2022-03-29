import cv2
import time
import numpy as np


def big_boy_func(value):
    pass


def create_kernel(img):
    col, row = img[1:-1,1:-1].shape

    for y in range(col):
        if y == 0:
            continue
        elif y == col:
            break

        for x in range(row):
            if x == 0:
                continue
            elif x == row:
                break

            ul = int(img[y-1, x-1])
            um = int(img[y-1, x])
            ur = int(img[y-1, x+1])
            cl = int(img[y, x-1])
            cm = int(img[y, x])
            cr = int(img[y, x+1])
            ll = int(img[y+1, x-1])
            lm = int(img[y+1, x])
            lr = int(img[y+1, x+1])
            mean_pixel_value = int((ul+um+ur+cl+cm+cr+ll+lm+lr)/9)
            img[y][x] = mean_pixel_value

    return img


def todo_1():
    cv2.namedWindow('mean')
    cv2.namedWindow('gaussian')
    cv2.namedWindow('median')
    cv2.createTrackbar('mask', 'mean', 1, 100, big_boy_func)
    cv2.createTrackbar('mask', 'gaussian', 1, 100, big_boy_func)
    cv2.createTrackbar('mask', 'median', 1, 100, big_boy_func)

    # noise = cv2.imread(r'pictures\lenna_noise.bmp')
    noise = cv2.imread(r'pictures\lenna_salt_and_pepper.bmp')

    while True:
        value_mean = cv2.getTrackbarPos('mask', 'mean')
        value_gauss = cv2.getTrackbarPos('mask', 'gaussian')
        value_median = cv2.getTrackbarPos('mask', 'median')

        if value_mean % 2 == 1:
            mean = cv2.blur(noise, (value_mean, value_mean))

        if value_gauss % 2 == 1:
            gauss = cv2.GaussianBlur(noise, (value_gauss, value_gauss), 0)

        if value_median % 2 == 1:
            median = cv2.medianBlur(noise, value_median)

        cv2.imshow('noise', noise)
        cv2.imshow('mean', mean)
        cv2.imshow('gaussian', gauss)
        cv2.imshow('median', median)

        if cv2.waitKey(100) == ord('q'):
            break

    cv2.destroyAllWindows()


def todo_2():
    cv2.namedWindow('dilate')
    cv2.namedWindow('erode')

    cv2.createTrackbar('mask', 'dilate', 1, 100, big_boy_func)
    cv2.createTrackbar('mask', 'erode', 1, 100, big_boy_func)

    # noise = cv2.imread(r'pictures\lenna_noise.bmp')
    noise = cv2.imread(r'pictures\lenna_salt_and_pepper.bmp')

    grey = cv2.cvtColor(noise, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(grey, 125, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('thresh', thresh)

    while True:
        dilate_mask = cv2.getTrackbarPos('mask', 'dilate')
        erode_mask = cv2.getTrackbarPos('mask', 'erode')

        dilate_mask_size = dilate_mask * 2 + 1
        erode_mask_size = erode_mask * 2 + 1

        dilate_element = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_mask_size, dilate_mask_size), (dilate_mask, dilate_mask))
        erode_element = cv2.getStructuringElement(cv2.MORPH_RECT, (erode_mask_size, erode_mask_size), (erode_mask, erode_mask))

        dilate = cv2.dilate(thresh, dilate_element)
        cv2.imshow('dilate', dilate)
        erode_dilate = cv2.erode(dilate, erode_element)
        cv2.imshow('erode_dilate', erode_dilate)

        erode = cv2.erode(thresh, erode_element)
        cv2.imshow('erode', erode)
        dilate_erode = cv2.dilate(erode, dilate_element)
        cv2.imshow('dilate_erode', dilate_erode)

        if cv2.waitKey(100) == ord('q'):
            break

    cv2.destroyAllWindows()


def todo_3():
    mcqueen = cv2.imread(r'pictures/mcqueen.jpg')
    mcqueen_resize = cv2.resize(mcqueen, (0, 0), fx=0.5, fy=0.5)
    mcqueen_grey = cv2.cvtColor(mcqueen_resize, cv2.COLOR_BGR2GRAY)
    mcqueen_stripes = mcqueen_grey.copy()
    mcqueen_stripes[:, ::3] = 255

    # my filter
    my_filter_time_start = time.perf_counter()
    mcqueen_stripes = create_kernel(mcqueen_stripes)

    my_filter_time_stop = time.perf_counter()

    # built-in filter - blur
    blur_timer_start = time.perf_counter()
    mcqueen_blur = cv2.blur(mcqueen_stripes, (3, 3))
    blur_timer_stop = time.perf_counter()

    # built-in filter - filter2d

    filterTWOd_timer_start = time.perf_counter()
    mcqueen_filterTWOd = cv2.filter2D(mcqueen_stripes, -1, (1))
    filterTWOd_timer_stop = time.perf_counter()

    # show outcome
    cv2.imshow(f'filter2D, time: {filterTWOd_timer_stop - filterTWOd_timer_start}', mcqueen_filterTWOd)
    cv2.imshow(f'blur, time: {blur_timer_stop - blur_timer_start}', mcqueen_blur)
    cv2.imshow(f'my filter, time: {my_filter_time_stop - my_filter_time_start}', mcqueen_stripes)
    cv2.waitKey()


def main():
    # todo_1()
    # todo_2()
    todo_3()


if __name__ == '__main__':
    main()
