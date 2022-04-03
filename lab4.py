import cv2
import numpy as np


ix, iy = -1, -1


def todo_1():
    def draw_rectangle(event, x, y, flag, param):
        global ix, iy

        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = x, y

        if event == cv2.EVENT_LBUTTONUP:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 3)
            cv2.imshow('img', img)

    img = cv2.imread(r'pictures\LOGO_PUT_VISION_LAB_MAIN.png')
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow('img')
    cv2.setMouseCallback('img', draw_rectangle)

    while True:

        cv2.imshow('img', img)
        if cv2.waitKey() == ord('q'):
           break

    cv2.destroyAllWindows()


def main():
    todo_1()


if __name__ == '__main__':
    main()
