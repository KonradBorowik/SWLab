import cv2
import numpy as np


# def isolate_white_blocks():
#     img = cv2.imread(r"pictures\project\img_014.jpg")
#     img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
#     img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#     thresh = cv2.threshold(img_gray, 168, 255, cv2.THRESH_BINARY)[1]
#
#     cv2.imshow('original', img_gray)
#     cv2.imshow('white blocks', thresh)
#     cv2.waitKey()


def concept_1():
    img = cv2.imread(r"pictures\project\img_018.jpg")
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray_blurred = cv2.GaussianBlur(img_gray, (3, 3), 0)
    img_canny = cv2.Canny(img_gray_blurred, 15, 30)

    kernel = np.ones((5, 5), np.uint8)
    dilate = cv2.dilate(img_canny, kernel)

    close = cv2.morphologyEx(dilate, cv2.MORPH_CROSS, (5, 5))

    # output = cv2.connectedComponentsWithStats(close, cv2.CV_32S)
    # (numLabels, labels, stats, centroids) = output

    contours, hierarchy = cv2.findContours(close, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img, contours, -1, (255, 0, 0), thickness=cv2.FILLED)

    cv2.imshow("original image", img)
    cv2.imshow("blurred", img_gray_blurred)
    cv2.imshow("canny", img_canny)
    cv2.imshow("close", close)
    cv2.imshow("dilate", dilate)
    cv2.waitKey()


if __name__ == "__main__":
    # isolate_white_blocks()
    concept_1()


# porównywanie kształtów przy pomocy momentów B)