import cv2
import numpy as np


def big_boy_func(value):
    pass


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



def main():
    # todo_1()
    # todo_2()
    todo_3()


if __name__ == '__main__':
    main()
