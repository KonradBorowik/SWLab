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


def todo_3():
    shapes = cv2.imread(r'pictures\shapes.jpg')
    shapes_gray = cv2.cvtColor(shapes, cv2.COLOR_BGR2GRAY)
    shapes_edges = cv2.Canny(shapes_gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(shapes_edges, 1, np.pi / 180, 200)

    lines_p = cv2.HoughLinesP(shapes_edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

    circles = cv2.HoughCircles(shapes_gray, cv2.HOUGH_GRADIENT, dp=1, minDist=20, param1=50, param2=40, minRadius=0, maxRadius=150)
    circles = np.uint16(np.around(circles))

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(shapes, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for line_p in lines_p:
        x1, y1, x2, y2 = line_p[0]
        cv2.line(shapes, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for circle in circles[0, :]:
        # print(circle)
        cv2.circle(shapes, (circle[0], circle[1]), circle[2], (0, 0, 0), 2)

    cv2.imshow('lines', shapes)
    cv2.waitKey()


def todo_4():
    platform = cv2.imread(r'pictures\drone_ship.jpg')

    gray_platform = cv2.cvtColor(platform, cv2.COLOR_BGR2GRAY)

    thresh_platform = cv2.threshold(gray_platform, 180, 255, cv2.THRESH_BINARY)[1]
    edges_platfrom = cv2.Canny(thresh_platform, 50, 150)

    lines_platform = cv2.HoughLinesP(edges_platfrom, 2, np.pi / 180, 120, minLineLength=180, maxLineGap=30)
    # print(lines_platform)

    apexes = []

    for line in lines_platform:
        x1, y1, x2, y2 = line[0]
        cv2.line(platform, (x1, y1), (x2, y2), (0, 255, 0), 5)

        apexes.append([x1, y1])
        apexes.append([x2, y2])

    cv2.line(platform, apexes[0], apexes[2], (0, 255, 0), 5)
    cv2.line(platform, apexes[1], apexes[3], (0, 255, 0), 5)

    cv2.imshow('gray', gray_platform)
    cv2.imshow('thresh', thresh_platform)
    cv2.imshow('edges', edges_platfrom)
    cv2.imshow('cel', platform)
    cv2.waitKey()


def todo_5():
    fruits = cv2.imread(r'pictures\fruit.jpg')
    fruits_hsv = cv2.cvtColor(fruits.copy(), cv2.COLOR_BGR2HSV)
    # fruits_gray = cv2.cvtColor(fruits.copy(), cv2.COLOR_BGR2GRAY)

    fruits_edges = cv2.Canny(fruits, 50, 150)

    circles = cv2.HoughCircles(fruits_edges, cv2.HOUGH_GRADIENT, dp=1, minDist=150, param1=50, param2=20, minRadius=100,
                               maxRadius=150)
    circles = np.uint16(np.around(circles))

    print(circles)

    lower_orange = np.array([10, 0, 0])
    upper_orange = np.array([20, 255, 255])

    lower_green = np.array([30, 0, 0])
    upper_green = np.array([50, 255, 255])

    for circle in circles[0, :]:
        if lower_orange[0] < (fruits_hsv[circle[1]+15][circle[0]+15])[0] < upper_orange[0]:
            cv2.circle(fruits, (circle[0], circle[1]), circle[2] + 10, (0, 165, 255), 5)
            cv2.putText(fruits, 'orange', (circle[0], circle[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)
        elif lower_green[0] < (fruits_hsv[circle[1]+15][circle[0]+15])[0] < upper_green[0]:
            cv2.circle(fruits, (circle[0], circle[1]), circle[2] + 10, (0, 255, 0), 5)
            cv2.putText(fruits, 'apple', (circle[0], circle[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 0, 0), 2)

    # print(f'bgr pixel value: {fruits[0][0]}')
    # print(f'hsv pixel value: {fruits_hsv[0][0]}')

    cv2.imshow('fruits', fruits)
    cv2.waitKey()


def todo_6():
    coins = cv2.imread(r'pictures\coins.jpg')

    canny_coins = cv2.Canny(coins, 20, 200)

    circles = cv2.HoughCircles(canny_coins, cv2.HOUGH_GRADIENT, dp=1, minDist=100, param1=70, param2=50, minRadius=10,
                               maxRadius=110)
    circles = np.uint16(np.around(circles))

    cash = 0

    for circle in circles[0, :]:
        # print(circle)
        if circle[2] > 100:
            cash += 1
            cv2.circle(coins, (circle[0], circle[1]), circle[2], (0, 0, 255), 4)
            cv2.putText(coins, '1PLN', (circle[0], circle[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
        elif 45 < circle[2] < 60:
            cash += 0.1
            cv2.circle(coins, (circle[0], circle[1]), circle[2], (255, 0, 0), 4)
            cv2.putText(coins, '0.1PLN', (circle[0], circle[1]), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

    cv2.putText(coins, f'cash = {round(cash, 1)} PLN', (45, 45), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
    cv2.imshow('coins', coins)
    cv2.waitKey()


def main():
    # todo_1()
    # todo_2()
    # todo_3()
    # todo_4()
    todo_5()
    # todo_6()


if __name__ == '__main__':
    main()
