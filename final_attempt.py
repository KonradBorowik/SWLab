import cv2
import numpy as np


def main(img):
    cv2.imshow('original', img)
    median = cv2.medianBlur(img, 9)
    cv2.imshow('median', median)

    # canny = cv2.Canny(median, 10, 59, cv2.THRESH_OTSU)
    # # cv2.imshow('canny', canny)
    #
    # dilate_kernel = np.ones((5, 5))
    # dilate_contours = cv2.morphologyEx(canny, cv2.MORPH_DILATE, dilate_kernel, iterations=2)
    # # cv2.imshow('dilate contours', dilate_contours)
    #
    # erode_kernel = np.ones((5, 5), dtype='uint8')
    # erode_thresh = cv2.morphologyEx(dilate_contours, cv2.MORPH_ERODE, erode_kernel, iterations=2)
    # cv2.imshow('erode', erode_thresh)
    #
    # img_to_clear = cv2.threshold(erode_thresh, 200, 255, cv2.THRESH_BINARY)[1]
    #
    # contours, hierarchy = cv2.findContours(img_to_clear, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # drawn_contours = cv2.drawContours(img.copy(), contours, -1, (255, 255, 255), -1)
    # cv2.imshow('contours', drawn_contours)

    # thresh_drawn_contours = cv2.threshold(drawn_contours.copy(), 250, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('thresh contours', thresh_drawn_contours)
    #
    #
    #
    # mask = np.zeros(img_to_clear.shape, dtype="uint8")
    #
    # for cnt in contours:
    #     if cv2.contourArea(cnt) > 1000:
    #         mask[cnt==cnt] = 255
    #
    # almost_only_legos = cv2.bitwise_and(img, img, mask=mask)
    # cv2.imshow('almost extracted legos', almost_only_legos)

    # # exclude table option
    # almost_only_legos_hsv = cv2.cvtColor(almost_only_legos.copy(), cv2.COLOR_BGR2HSV)
    # erode_kernel_for_colors = np.ones([3, 3])
    # # white
    # white_lower = np.array([0, 0, 160])
    # white_upper = np.array([179, 25, 255])
    #
    # white = cv2.inRange(almost_only_legos_hsv, white_lower, white_upper)
    # white = cv2.dilate(white, erode_kernel_for_colors)
    # # join = cv2.bitwise_and(almost_only_legos, almost_only_legos, mask=white)
    # # cv2.imshow('only white', white)
    #
    #
    # # red
    # red_low_lower = np.array([0, 60, 65])
    # red_low_upper = np.array([10, 255, 255])
    # red_up_lower = np.array([160, 80, 85])
    # red_up_upper = np.array([179, 255, 255])
    #
    # red_low_mask = cv2.inRange(almost_only_legos_hsv, red_low_lower, red_low_upper)
    # red_up_mask = cv2.inRange(almost_only_legos_hsv, red_up_lower, red_up_upper)
    #
    # red_mask = red_low_mask + red_up_mask
    # red = cv2.dilate(red_mask, erode_kernel_for_colors)
    # # cv2.imshow('red', red)
    #
    # # yellow
    # yellow_lower = np.array([20, 100, 90])
    # yellow_upper = np.array([35, 255, 255])
    #
    # yellow_mask = cv2.inRange(almost_only_legos_hsv, yellow_lower, yellow_upper)
    # yellow = cv2.dilate(yellow_mask, erode_kernel_for_colors)
    # # yellow = cv2.bitwise_and(almost_only_legos, almost_only_legos, mask=yellow_mask)
    # # cv2.imshow('yellow', yellow)
    #
    # # green
    # green_lower = np.array([60, 50, 25])
    # green_upper = np.array([90, 255, 255])
    #
    # green_mask = cv2.inRange(almost_only_legos_hsv, green_lower, green_upper)
    # green = cv2.dilate(green_mask, erode_kernel_for_colors)
    # # green = cv2.bitwise_and(almost_only_legos, almost_only_legos, mask=green_mask)
    # # cv2.imshow('green', green)
    #
    # # blue
    # blue_lower = np.array([90, 25, 15])
    # blue_upper = np.array([130, 255, 255])
    #
    # blue_mask = cv2.inRange(almost_only_legos_hsv, blue_lower, blue_upper)
    # blue = cv2.dilate(blue_mask, erode_kernel_for_colors)
    # # blue = cv2.bitwise_and(almost_only_legos, almost_only_legos, mask=blue_mask)
    # # cv2.imshow('blue', blue)
    #
    # all_colors = white + yellow + red + green + blue
    # # cv2.imshow('all', all_colors)
    # joined_colors = cv2.bitwise_and(almost_only_legos, almost_only_legos, mask=all_colors)
    # # cv2.imshow('fully extracted legos', joined_colors)
    #
    # reshape_for_quant = joined_colors.reshape
    #
    # gray = cv2.cvtColor(joined_colors, cv2.COLOR_BGR2GRAY)
    # # cv2.imshow('gray', gray)
    #
    # thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
    # # cv2.imshow('thresh', thresh)
    #
    # sharpened = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, (5, 5))
    # # cv2.imshow('sharpened', sharpened)

    cv2.waitKey()


if __name__ == "__main__":
    for i in range(5):
        sample = cv2.imread(f'pictures/project/sample/img_{i + 1:03}.jpg')
        resized_sample = cv2.resize(sample, (0, 0), fx=0.15, fy=0.15)
        main(resized_sample)
