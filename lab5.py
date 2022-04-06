import cv2
import numpy as np


def big_boy_func(x):
    pass


def todo_1():
    lena_gray = cv2.imread(r'pictures\lena.jpg', cv2.IMREAD_GRAYSCALE)

    kernel_prewitt_x = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]])
    kernel_prewitt_y = np.array([[1, 1, 1], [0, 0, 0],[-1, -1, -1]])

    kernel_sobel_x = np.array([[1, 0, -1],[2, 0, -2],[1, 0, -1]])
    kernel_sobel_y = np.array([[1, 2, 1],[0, 0, 0],[-1, -2, -1]])

    prewitt_x = cv2.filter2D(lena_gray, cv2.CV_32F, kernel_prewitt_x) / 3
    prewitt_y = cv2.filter2D(lena_gray, cv2.CV_32F, kernel_prewitt_y) / 3

    sobel_x = cv2.filter2D(lena_gray, cv2.CV_32F, kernel_sobel_x) / 4
    sobel_y = cv2.filter2D(lena_gray, cv2.CV_32F, kernel_sobel_y) / 4

    # moduł gradientu
    sobel_x_64f = cv2.filter2D(lena_gray, cv2.CV_64F, kernel_sobel_x)
    abs_sobel_x_64f = np.absolute(sobel_x_64f)
    sobel_x_2 = np.uint8(abs_sobel_x_64f)

    cv2.imshow('lena', sobel_x)
    cv2.imshow('lena 2', sobel_x_2)
    cv2.waitKey()


def todo_2():
    cv2.namedWindow('lenna canny')
    lena_gray = cv2.imread(r'pictures\lena.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.createTrackbar('max_val', 'lenna canny', 0, 255, big_boy_func)
    cv2.createTrackbar('min_val', 'lenna canny', 0, 255, big_boy_func)

    while True:
        max_thresh = cv2.getTrackbarPos('max_val', 'lenna canny')
        min_thresh = cv2.getTrackbarPos('min_val', 'lenna canny')

        lena_canny = cv2.Canny(lena_gray, min_thresh, max_thresh)

        cv2.imshow('gray lenna', lena_gray)
        cv2.imshow('lenna canny', lena_canny)

        if cv2.waitKey(50) == ord('q'):
            break

    cv2.destroyAllWindows()



def main():
    todo_1()
    # todo_2()


if __name__ == '__main__':
    main()
