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


# def brighten_dark_spots():
#     img = cv2.imread(r"pictures\project\img_012.jpg")
#     img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
#
#     cv2.imshow("original", img)
#     cv2.waitKey()


# def brighten_dark_spots(img):
#     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     for y in range(img_hsv.shape[0]):
#         for x in range(img_hsv.shape[1]):
#             if img_hsv[y, x][2] < 80:
#                 img_hsv[y, x][2] = 150
#
#     img_bgr = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
#
#     return img_bgr
from scipy.stats import stats


def concept_1():
    # read and resize picture
    img = cv2.imread(r"pictures\project\img_014.jpg")
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply blur (somehow edges are more likely to be detected)
    img_gray_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # get outlines of legos
    img_canny = cv2.Canny(img_gray_blurred, 15, 20, cv2.THRESH_OTSU)
    # close outlines
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(img_canny, kernel_dilate)

    # close = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, (5, 5))

    # copy of main picture to work with
    image_with_contours = img.copy()
    # get and draw contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_with_contours, contours, -1, (255, 255, 255), -1)

    # step 2: obtain only legos
    thresh_drawn_contours = cv2.threshold(image_with_contours, 254, 255, cv2.THRESH_BINARY)[1]
    # erode few times to remove noise
    erode = cv2.morphologyEx(thresh_drawn_contours, cv2.MORPH_ERODE, (3, 3), iterations=10)

    # (numLabels, labels, stats, centroids) = cv2.connectedComponentsWithStats(erode, cv2.CV_32S)

    # print outcome
    cv2.imshow("original image", img)
    # cv2.imshow("blurred", img_gray_blurred)
    # cv2.imshow("canny", img_canny)
    # cv2.imshow("close", close)
    # cv2.imshow("dilate", dilate)
    cv2.imshow("drawn contours", image_with_contours)
    cv2.imshow("threshold", thresh_drawn_contours)
    cv2.imshow("gradient", erode)
    cv2.waitKey()


if __name__ == "__main__":
    # isolate_white_blocks()
    concept_1()
    # brighten_dark_spots()


# porównywanie kształtów przy pomocy momentów B)