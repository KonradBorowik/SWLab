import cv2
import numpy as np
from skimage import measure
import time
import matplotlib.pyplot as plt


if __name__ == "__main__":
    start = time.time()
    # read and resize picture
    img = cv2.imread(r"pictures\project\img_018.jpg")
    resized_img = cv2.resize(img, (0, 0), fx=0.2, fy=0.2)
    # cv2.imshow('original', resized_img)
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
        if num_pixels > 300:
            mask = cv2.add(mask, label_mask)

    almost_only_legos = cv2.bitwise_and(resized_img, resized_img, mask=mask)
    # cv2.imshow('almost extracted legos', almost_only_legos)

    # exclude table option
    almost_only_legos_hsv = cv2.cvtColor(almost_only_legos.copy(), cv2.COLOR_BGR2HSV)
    erode_kernel_for_colors = np.ones([3, 3])
    # white
    white_lower = np.array([0, 0, 160])
    white_upper = np.array([179, 25, 255])

    white = cv2.inRange(almost_only_legos_hsv, white_lower, white_upper)
    white = cv2.dilate(white, erode_kernel_for_colors)
    # join = cv2.bitwise_and(almost_only_legos, almost_only_legos, mask=white)
    # cv2.imshow('only white', white)


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
    yellow_lower = np.array([20, 100, 90])
    yellow_upper = np.array([35, 255, 255])

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
    blue_lower = np.array([90, 25, 15])
    blue_upper = np.array([130, 255, 255])

    blue_mask = cv2.inRange(almost_only_legos_hsv, blue_lower, blue_upper)
    blue = cv2.dilate(blue_mask, erode_kernel_for_colors)
    # blue = cv2.bitwise_and(almost_only_legos, almost_only_legos, mask=blue_mask)
    # cv2.imshow('blue', blue)

    all_colors = white + yellow + red + green + blue
    # cv2.imshow('all', all_colors)
    joined_colors = cv2.bitwise_and(almost_only_legos, almost_only_legos, mask=all_colors)
    # cv2.imshow('fully extracted legos', joined_colors)

    reshape_for_quant = joined_colors.reshape

    gray = cv2.cvtColor(joined_colors, cv2.COLOR_BGR2GRAY)
    # cv2.imshow('gray', gray)

    thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow('thresh', thresh)

    sharpened = cv2.morphologyEx(thresh, cv2.MORPH_DILATE, (5, 5))
    # cv2.imshow('sharpened', sharpened)

    contours, hierarchy = cv2.findContours(sharpened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # cv2.drawContours(resized_img, contours[3], -1, (110, 110, 110), thickness=13)
    # cv2.imshow('original', resized_img)

    for cnt in contours:
        if cv2.contourArea(cnt) < 1000:
            continue

    resized_hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    img_to_draw_on = resized_img.copy()

    color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

    with open('test.npy', 'rb') as f:
        sample = np.load(f, allow_pickle=True)

    red_count, green_count, blue_count, white_count, yellow_count, mix_count = 0, 0, 0, 0, 0, 0
    shape_one, shape_two, shape_three, shape_four, shape_five = 0, 0, 0, 0, 0

    for t_cnt in contours:
        t_cnt_area = cv2.contourArea(t_cnt)
        if t_cnt_area < 1000:
            continue
        match = []

        cnt_mask = np.zeros(resized_img.shape[:2], np.uint8)
        single_cnt = cv2.drawContours(cnt_mask, [t_cnt], -1, (255, 255, 255), -1)
        mask_and_img = cv2.bitwise_and(resized_img, resized_img, mask=single_cnt)

        mean_pixel_val = cv2.mean(resized_hsv_img, mask=single_cnt)
        # cv2.imshow('cnt', mask_and_img)
        # cv2.waitKey()
        mask_and_img_hsv = cv2.cvtColor(mask_and_img, cv2.COLOR_)
        hist_h = cv2.calcHist([mask_and_img_hsv], [0], None, [256], [1, 256])

        # hist_s = cv2.calcHist([mask_and_img_hsv[:,:,1]], [0], None, [256], [1, 256])
        # hist_v = cv2.calcHist([mask_and_img_hsv[:,:,2]], [0], None, [256], [1, 256])
        plt.plot(hist_h, color='r', label="h")
        # plt.plot(hist_s, color='g', label="s")
        # plt.plot(hist_v, color='b', label="v")

        plt.show()

        # important_hue = []
        # for i in range(len(hist_h)):
        #     if hist_h[i] > 120:
        #         if not (hist_h[i-1] > 120):
        #             important_hue.append(i)
        #
        # # print(important_hue)
        # max_val = max(hist_h)
        # if len(important_hue) > 1:
        #     block_color = 6 # mix
        # elif len(important_hue) > 2:
        #     block_color = 4 # white
        # else:
        #     if 40 < list(hist_h).index(max_val):
        #         block_color = 1 # red
        #     elif -1 < list(hist_h).index(max_val) < 5:
        #         block_color = 1 # red
        #     elif 4 < list(hist_h).index(max_val) < 9:
        #         block_color = 5 # yellow
        #     elif 16 < list(hist_h).index(max_val) < 20:
        #         block_color = 2 # green
        #     elif 24 < list(hist_h).index(max_val) < 30:
        #         block_color = 3 # blue

        x, y, w, h = cv2.boundingRect(t_cnt)
        best_match = []
        for s_cnts in sample:
            for s_cnt in s_cnts:
                match.append(cv2.matchShapes(t_cnt, s_cnt, cv2.CONTOURS_MATCH_I1, 0.0))

            best_match.append(min(match))
            match = []

        # print(f'sample{sam}: {match}; color: {block_color}')

        min_value = min(best_match)
        print(min_value)
        if best_match[best_match.index(min_value)] < 0.1:
            # important_hue = []
            # for i in range(len(hist_h)):
            #     if hist_h[i] > 120:
            #         if not (hist_h[i - 1] > 120):
            #             important_hue.append(i)

            hue = 0
            print(len(t_cnt))
            for pixel in t_cnt:
                # hue += mask_and_img_hsv[pixel[0][1], pixel[0][0], 0]
                hue += 1

            # mean_hue = hue / (len(t_cnt) - 1)
            print(hue)

            # # print(important_hue)
            # max_val = max(hist_h)
            # # if len(important_hue) > 3:
            # #     block_color = 4  # white
            # #     white_count += 1
            # if len(important_hue) > 1:
            #     block_color = 6  # mix
            #     mix_count += 1
            # else:
            #     if 40 < list(hist_h).index(max_val):
            #         block_color = 1  # red
            #         red_count += 1
            #     elif -1 < list(hist_h).index(max_val) < 5:
            #         block_color = 1  # red
            #         red_count += 1
            #     elif 4 < list(hist_h).index(max_val) < 9:
            #         block_color = 5  # yellow
            #         yellow_count += 1
            #     elif 16 < list(hist_h).index(max_val) < 20:
            #         block_color = 2  # green
            #         green_count += 1
            #     elif 24 < list(hist_h).index(max_val) < 30:
            #         block_color = 3  # blue
            #         blue_count += 1
            #     else:
            #         block_color = 4
            #         white_count += 1
            block_color = 1
            cv2.drawContours(img_to_draw_on, t_cnt, -1, color[best_match.index(min_value)], 6)
            cv2.putText(img_to_draw_on, f'{block_color}', (x + 20, y + 20), cv2.FONT_HERSHEY_PLAIN, 4, 2)

            shape_type = best_match.index(min_value)
            if shape_type == 0:
                shape_one += 1
            elif shape_type == 1:
                shape_two += 1
            elif shape_type == 3:
                shape_three += 1
            elif shape_type == 4:
                shape_four += 1
            elif shape_type == 2:
                shape_five += 1
            end = time.time()
        cv2.imshow('test', img_to_draw_on)
        cv2.waitKey()

    end = time.time()
    print(f'elapsed time: {end - start}')

    outcome = [shape_one, shape_two, shape_three, shape_four, shape_five, red_count, green_count, blue_count,
               white_count, yellow_count, mix_count]
    print(outcome)

    cv2.waitKey()
    cv2.destroyAllWindows()
