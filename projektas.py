import cv2
import numpy as np
from skimage import measure


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



def big_boy_func(x):
    pass


def concept2(img):
    cv2.namedWindow('ctrl', cv2.WINDOW_GUI_EXPANDED)
    # img = cv2.imread(r'pictures\project\img_014.jpg')
    resized_img = cv2.resize(img, (0, 0), fy=0.2, fx=0.2)
    # gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    # inv = cv2.bitwise_not(gray)
    #
    # cv2.imshow('gray', gray)
    # cv2.imshow('inv', inv)
    # cv2.waitKey()

    # median = cv2.medianBlur(resized_img, 9)
    # cv2.imshow('median', median)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    cv2.createTrackbar('hmin', 'ctrl', 0, 179, big_boy_func)
    cv2.createTrackbar('smin', 'ctrl', 0, 255, big_boy_func)
    cv2.createTrackbar('vmin', 'ctrl', 165, 255, big_boy_func)
    cv2.createTrackbar('hmax', 'ctrl', 179, 179, big_boy_func)
    cv2.createTrackbar('smax', 'ctrl', 50, 255, big_boy_func)
    cv2.createTrackbar('vmax', 'ctrl', 255, 255, big_boy_func)

    # # red
    # red_low_lower = np.array([0, 60, 65])
    # red_low_upper = np.array([10, 255, 255])
    # red_up_lower = np.array([160, 80, 85])
    # red_up_upper = np.array([179, 255, 255])
    #
    # mask_low = cv2.inRange(img, red_low_lower, red_low_upper)
    # mask_up = cv2.inRange(img, red_up_lower, red_up_upper)
    #
    # join_low = cv2.bitwise_and(img, img, mask=mask_low)
    # join_up = cv2.bitwise_and(img, img, mask=mask_up)
    # #
    # join = cv2.bitwise_or(join_up, join_low)
    #
    # # show image
    # img_joined = cv2.cvtColor(join, cv2.COLOR_HSV2BGR)
    # cv2.imshow('alk;sdjgh', img_joined)
    # cv2.waitKey()
    #
    while True:
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        hmin = cv2.getTrackbarPos('hmin', 'ctrl')
        hmax = cv2.getTrackbarPos('hmax', 'ctrl')
        smin = cv2.getTrackbarPos('smin', 'ctrl')
        smax = cv2.getTrackbarPos('smax', 'ctrl')
        vmin = cv2.getTrackbarPos('vmin', 'ctrl')
        vmax = cv2.getTrackbarPos('vmax', 'ctrl')

        lower = np.array([hmin, smin, vmin])
        upper = np.array([hmax, smax, vmax])
        # lower = np.array([0, 0, 161])
        # upper = np.array([179, 33, 190])
        mask = cv2.inRange(img_hsv, lower, upper)
        # img_and = cv2.bitwise_and(img, img, mask=mask)

        # white
        # canny = cv2.Canny(img_and, 30, 50)

        join = cv2.bitwise_and(img, img, mask=mask)

        # cv2.imshow("canny", canny)
        # cv2.imshow("join", join)

        #
        # img_and = cv2.cvtColor(img_and, cv2.COLOR_HSV2BGR)
        # cv2.imshow('aaa', img_and)
        # mask = cv2.cvtColor(mask, cv2.COLOR_HSV2BGR)
        cv2.imshow('thresh', join)

        if cv2.waitKey(50) == ord('q'):
            break

    cv2.destroyAllWindows()


def extract_legos():
    from scipy.stats import stats
    # read and resize picture
    img = cv2.imread(r"pictures\project\img_012.jpg")
    img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # apply blur (somehow edges are more likely to be detected)
    img_gray_blurred = cv2.GaussianBlur(img, (5, 5), 0)
    # get outlines of legos
    img_canny = cv2.Canny(img_gray_blurred, 15, 17, cv2.THRESH_OTSU)
    # close outlines
    kernel_dilate = np.ones((3, 3), np.uint8)
    dilate = cv2.dilate(img_canny, kernel_dilate)

    # copy of main picture to work with
    image_with_contours = img.copy()
    # get and draw contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image_with_contours, contours, -1, (255, 255, 255), -1)

    # step 2: obtain only legos
    thresh_drawn_contours = cv2.threshold(image_with_contours, 254, 255, cv2.THRESH_BINARY)[1]
    # erode few times to remove noise
    erode = cv2.morphologyEx(thresh_drawn_contours, cv2.MORPH_ERODE, (3, 3), iterations=3)

    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    img_to_clear = cv2.cvtColor(erode.copy(), cv2.COLOR_BGR2GRAY)
    img_to_clear = cv2.threshold(img_to_clear, 200, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('img to clear', img_to_clear)
    labels = measure.label(img_to_clear, connectivity=2, background=0)
    mask = np.zeros(img_to_clear.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        label_mask = np.zeros(img_to_clear.shape, dtype="uint8")
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "legos"
        if num_pixels > 1000:
            mask = cv2.add(mask, label_mask)

    join = cv2.bitwise_and(img, img, mask=mask)

    # print outcome
    # cv2.imshow("original image", img)
    # cv2.imshow("blurred", img_gray_blurred)
    # cv2.imshow("canny", img_canny)
    # # cv2.imshow("close", close)
    # cv2.imshow("dilate", dilate)
    # cv2.imshow("drawn contours", image_with_contours)
    # cv2.imshow("threshold", thresh_drawn_contours)
    # cv2.imshow("gradient", erode)
    cv2.imshow("img to clear", mask)
    cv2.imshow('join', join)
    cv2.waitKey()

    return join, mask


def exclude_table(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    upper_thresh = np.array([1, 0, 75])
    lower_thresh = np.array([179, 90, 140])

    table = cv2.inRange(img_hsv, upper_thresh, lower_thresh)

    table_to_remove = cv2.bitwise_or(img, img, mask=table)

    only_legos = img - table_to_remove

    thresh_only_legos = cv2.threshold(only_legos, 10, 255, cv2.THRESH_BINARY)[1]

    erode_thresh = cv2.erode(thresh_only_legos, (5, 5), iterations=3)

    cv2.imshow('test', only_legos)
    cv2.imshow('asdfg', thresh_only_legos)
    cv2.imshow('erode thresh', erode_thresh)


def concept3():
    original_img = cv2.imread(r'pictures/project/img_014.jpg')
    resized_img = cv2.resize(original_img, (0, 0), fx=0.2, fy=0.2)
    img_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    div = 20
    img = img_gray // div * div + div // 2

    canny = cv2.Canny(img, 10, 25, cv2.THRESH_OTSU)

    cv2.imshow('original', resized_img)
    cv2.imshow('canny', canny)
    # cv2.imshow('draw around every block', img)
    cv2.waitKey()


def trackbars(img):
    cv2.namedWindow('ctrl')
    cv2.createTrackbar('val1', 'ctrl', 0, 255, big_boy_func)
    cv2.createTrackbar('val2', 'ctrl', 0, 255, big_boy_func)

    while True:
        val1 = cv2.getTrackbarPos('val1', 'ctrl')
        val2 = cv2.getTrackbarPos('val2', 'ctrl')

        img_canny = cv2.Canny(img, val1, val2)
        cv2.imshow('img canny test', img_canny)

        if cv2.waitKey(50) == ord('q'):
            break


def concept4(original_img):
    # original_img = cv2.imread(r'pictures/project/img_005.jpg')
    resized_img = cv2.resize(original_img, (0, 0), fx=0.2, fy=0.2)
    cv2.imshow('resized', resized_img)

    median = cv2.medianBlur(resized_img, 9)
    # cv2.imshow('median', median)

    # trackbars(median)

    canny = cv2.Canny(median, 10, 59, cv2.THRESH_OTSU)
    # cv2.imshow('canny', canny)

    dilate_kernel = np.ones((5, 5))
    dilate_contours = cv2.morphologyEx(canny, cv2.MORPH_DILATE, dilate_kernel, iterations=2)
    # cv2.imshow('dilate contours', dilate_contours)

    contours, hierarchy = cv2.findContours(dilate_contours, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    drawn_contours = cv2.drawContours(resized_img.copy(), contours, -1, (255, 255, 255), -1)
    # cv2.imshow('contours', drawn_contours)

    # thresh_drawn_contours = cv2.threshold(drawn_contours.copy(), 250, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('thresh contours', thresh_drawn_contours)

    erode_kernel = np.ones((5, 5), dtype='uint8')
    erode_thresh = cv2.morphologyEx(drawn_contours, cv2.MORPH_ERODE, erode_kernel, iterations=2)
    # cv2.imshow('erode', erode_thresh)

    img_to_clear = cv2.cvtColor(erode_thresh.copy(), cv2.COLOR_BGR2GRAY)
    img_to_clear = cv2.threshold(img_to_clear, 200, 255, cv2.THRESH_BINARY)[1]

    labels = measure.label(img_to_clear, connectivity=2, background=0)
    mask = np.zeros(img_to_clear.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):
        # if this is the background label, ignore it
        if label == 0:
            continue
        # otherwise, construct the label mask and count the
        # number of pixels
        label_mask = np.zeros(img_to_clear.shape, dtype="uint8")
        label_mask[labels == label] = 255
        num_pixels = cv2.countNonZero(label_mask)
        # if the number of pixels in the component is sufficiently
        # large, then add it to our mask of "legos"
        if num_pixels > 600:
            mask = cv2.add(mask, label_mask)

    almost_only_legos = cv2.bitwise_and(resized_img, resized_img, mask=mask)
    cv2.imshow('almost extracted legos', almost_only_legos)

    # exclude table option
    almost_only_legos_hsv = cv2.cvtColor(almost_only_legos.copy(), cv2.COLOR_BGR2HSV)
    erode_kernel_for_colors = np.ones([3, 3])
    # white
    white_lower = np.array([0, 0, 165])
    white_upper = np.array([179, 50, 255])

    white = cv2.inRange(almost_only_legos_hsv, white_lower, white_upper)
    white = cv2.dilate(white, erode_kernel_for_colors)
    # join = cv2.bitwise_and(almost_only_legos, almost_only_legos, mask=white)
    # cv2.imshow('only white', join)

    # red
    red_low_lower = np.array([0, 60, 65])
    red_low_upper = np.array([10, 255, 255])
    red_up_lower = np.array([160, 80, 85])
    red_up_upper = np.array([179, 255, 255])

    red_low_mask = cv2.inRange(almost_only_legos_hsv, red_low_lower, red_low_upper)
    red_up_mask = cv2.inRange(almost_only_legos_hsv, red_up_lower, red_up_upper)

    red_mask = red_low_mask + red_up_mask
    red = cv2.dilate(red_mask, erode_kernel_for_colors)
    # cv2.imshow('red', red)

    # yellow
    yellow_lower = np.array([10, 80, 90])
    yellow_upper = np.array([30, 255, 185])

    yellow_mask = cv2.inRange(almost_only_legos_hsv, yellow_lower, yellow_upper)
    yellow = cv2.dilate(yellow_mask, erode_kernel_for_colors)
    # yellow = cv2.bitwise_and(almost_only_legos, almost_only_legos, mask=yellow_mask)
    # cv2.imshow('yellow', yellow)

    # green
    green_lower = np.array([60, 50, 25])
    green_upper = np.array([90, 255, 255])

    green_mask = cv2.inRange(almost_only_legos_hsv, green_lower, green_upper)
    green = cv2.dilate(green_mask, erode_kernel_for_colors)
    # green = cv2.bitwise_and(almost_only_legos, almost_only_legos, mask=green_mask)
    # cv2.imshow('green', green)

    # blue
    blue_lower = np.array([90, 50, 25])
    blue_upper = np.array([130, 255, 255])

    blue_mask = cv2.inRange(almost_only_legos_hsv, blue_lower, blue_upper)
    blue = cv2.dilate(blue_mask, erode_kernel_for_colors)
    # blue = cv2.bitwise_and(almost_only_legos, almost_only_legos, mask=blue_mask)
    # cv2.imshow('blue', blue)

    all_colors = white + yellow + red + green + blue
    joined_colors = cv2.bitwise_and(almost_only_legos, almost_only_legos, mask=all_colors)
    cv2.imshow('fully extracted legos', joined_colors)

    cv2.waitKey()

if __name__ == "__main__":
    # img, mask = extract_legos()
    #
    # exclude_table(img)
    # # brighten_dark_spots()
    # concept2()

    # concept3()

    for i in range(17):
        img = cv2.imread(f'pictures/project/img_{i+1:03}.jpg')
        concept4(img)


# porównywanie kształtów przy pomocy momentów B)