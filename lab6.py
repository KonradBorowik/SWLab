import cv2
import numpy as np


def big_boy_function(*popo):
    pass


def todo_1():
    not_bad = cv2.imread(r'pictures\not_bad.jpg')
    resize_not_bad = cv2.resize(not_bad, (0, 0), fx=0.2, fy=0.2)
    gray_not_bad = cv2.cvtColor(resize_not_bad, cv2.COLOR_BGR2GRAY)

    thresh_not_bad = cv2.threshold(gray_not_bad, 50, 255, cv2.THRESH_BINARY_INV)[1]
    kernel = np.ones((7, 7))
    thresh_not_bad = cv2.dilate(thresh_not_bad, kernel)
    thresh_not_bad = cv2.erode(thresh_not_bad, kernel)

    contours, hierarchy = cv2.findContours(thresh_not_bad, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(resize_not_bad, contours, -1, (0,255,0), 3)

    moments = []
    for cnt in contours:
        moments.append(cv2.moments(cnt))

    # print(mu)
    centers = []
    i = 0
    for mu in moments:
        mc = [mu['m10'] / mu['m00'], mu['m01'] / mu['m00']]
        cv2.circle(resize_not_bad, ((int(mc[0])), int(mc[1])), 4, (0, 0, 255), 3)
        i += 1
        centers.append(mc)

    points = np.float32([[resize_not_bad.shape[1], resize_not_bad.shape[0]], [0, resize_not_bad.shape[0]], [resize_not_bad.shape[1], 0], [0, 0]])
    centers = np.float32(centers)
    print(resize_not_bad.shape)
    transform = cv2.getPerspectiveTransform(centers, points)
    straightened = cv2.warpPerspective(resize_not_bad, transform, (653, 490))

    cv2.imshow('color', resize_not_bad)
    cv2.imshow('gray', gray_not_bad)
    cv2.imshow('thresh', thresh_not_bad)
    cv2.imshow('document', straightened)
    cv2.waitKey()


def todo_2():
    kubus = cv2.imread(r'pictures\kubus.jpg')
    face_kubus = kubus[50:210, 250:390]

    cv2.namedWindow('kubus')
    cv2.createTrackbar('method', 'kubus', 0, 5, big_boy_function)
    methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR, cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]

    while True:
        chosen_method = cv2.getTrackbarPos('method', 'kubus')

        res = cv2.matchTemplate(kubus, face_kubus, methods[chosen_method])
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if methods in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
            bottom_right = (top_left[0] + face_kubus.shape[1], top_left[1] + face_kubus.shape[0])

        cv2.rectangle(kubus, top_left, bottom_right, 255, 2)

        cv2.imshow('kubus', kubus)
        cv2.imshow('template', face_kubus)

        if cv2.waitKey(100) == ord('q'):
            break


def main():
    # todo_1()
    todo_2()


if __name__ == "__main__":
    main()