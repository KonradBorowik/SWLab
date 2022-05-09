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


def brighten_dark_spots():
    img = cv2.imread(r"pictures\project\img_012.jpg")
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)



    cv2.imshow("original", img)
    cv2.waitKey()


def concept_1():
    img = cv2.imread(r"pictures\project\img_016.jpg")
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_gray_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    img_canny = cv2.Canny(img_gray_blurred, 15, 20, cv2.THRESH_OTSU)

    kernel_dilate = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(img_canny, kernel_dilate)

    # close = cv2.morphologyEx(dilate, cv2.MORPH_OPEN, (5, 5))

    # output = cv2.connectedComponentsWithStats(erode, cv2.CV_32S)
    # (numLabels, labels, stats, centroids) = output

    image_for_contours = img.copy()

    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_for_contours, contours, -1, (255, 255, 255), -1)

    # step 2: obtain only legos
    


    cv2.imshow("original image", img)
    cv2.imshow("blurred", img_gray_blurred)
    cv2.imshow("canny", img_canny)
    # cv2.imshow("close", close)
    cv2.imshow("dilate", dilate)
    cv2.imshow("drawn contours", image_for_contours)
    cv2.waitKey()


if __name__ == "__main__":
    # isolate_white_blocks()
    concept_1()
    # brighten_dark_spots()


# porównywanie kształtów przy pomocy momentów B)