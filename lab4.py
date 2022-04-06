import cv2
import numpy as np
from matplotlib import pyplot as plt


ix, iy = -1, -1
# this one is for todo_2()
points1 = []
#this one is for todo_4()
pts = []
i = 0
#todo_5()
mcqueen_pos = []


def todo_1():
    def draw_rectangle(event, x, y, flag, param):
        global ix, iy

        if event == cv2.EVENT_LBUTTONDOWN:
            ix, iy = x, y
        elif event == cv2.EVENT_RBUTTONDOWN:
            ix, iy = x, y

        if event == cv2.EVENT_LBUTTONUP:
            cv2.rectangle(img, (ix, iy), (x, y), (0, 0, 255), 3)
            cv2.imshow('img', img)
        elif event == cv2.EVENT_RBUTTONUP:
            cv2.circle(img, (ix, iy), 5, (0, 255, 0), 3)
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


def todo_2():
    def straightener(event, x, y, flag, param):
        global points1

        if event == cv2.EVENT_LBUTTONDOWN:
            points1.append((x, y))
            print(f'click {points1}')

        if len(points1) == 4:
            points1 = np.float32(points1)
            print(points1)
            M = cv2.getPerspectiveTransform(points1, points2)
            dst = cv2.warpPerspective(img, M, (300, 300))

            cv2.imshow('straightened', dst)
            if cv2.waitKey() == ord('d'):
                cv2.destroyWindow('straightened')
                points1 = []

    points2 = np.float32([[0, 0], [300, 0], [0, 300], [300, 300]])

    img = cv2.imread(r'pictures\road.jpg')
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3, )
    cv2.namedWindow('original')
    cv2.setMouseCallback('original', straightener)

    while True:
        cv2.imshow('original', img)

        if cv2.waitKey() == ord('q'):
            break

    cv2.destroyAllWindows()


def todo_3():
    img = cv2.imread(r'pictures\lena.jpg')
    # img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img_hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

    colors = ('b', 'g', 'r')

    for i, col in enumerate(colors):
        img_hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(img_hist, color=col)
        plt.xlim([0, 256])

    plt.plot(gray_img_hist, color='black')

    equ_gray_img = cv2.equalizeHist(gray_img)
    res_equ_gray_img = np.hstack((gray_img, equ_gray_img))

    equ_gray_img_hist = cv2.calcHist([equ_gray_img], [0], None, [256], [0, 256])

    plt.plot(equ_gray_img_hist, color='orange')
    plt.show()
    cv2.imshow('equed', res_equ_gray_img)
    cv2.waitKey()


def todo_4():
    def cropper(event, x, y, flag, param):
        global pts, i

        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append([x, y])
            print(f'{pts}')

        if len(pts) == 2:
            cropped_image = img[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]]

            cropped_image[:, :, 0] = 0
            cropped_image[:, :, 2] = 0
            cv2.imshow('cutout', cropped_image)

            g_thresh = cv2.threshold(cropped_image, 100, 255, cv2.THRESH_BINARY)[1]
            # new_img = img.copy()
            img[pts[0][1]:pts[1][1], pts[0][0]:pts[1][0]] = g_thresh
            g_t_img = img

            cv2.putText(img, f'{i}', (pts[0][0], pts[0][1]), cv2.FONT_HERSHEY_PLAIN, 5, (255,0,0))
            cv2.imshow('image', img)

            pts = []
            i += 1
            cv2.waitKey()

    img = cv2.imread(r'pictures\lena.jpg')
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', cropper)

    while True:
        cv2.imshow('image', img)

        if cv2.waitKey() == ord('q'):
            break


def todo_5():
    mcqueen_pos = []
    mcqueen = cv2.imread(r'pictures/mcqueen.jpg')

    def input_mcqueen(event, x, y, bla, ble):
        # global mcqueen_pos
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(mcqueen_pos) < 4:
                mcqueen_pos.append([x, y])
                print(mcqueen_pos)
                print(len(mcqueen_pos))

    img = cv2.imread(r'pictures\road.jpg')
    img = cv2.resize(img, (0, 0), fx=0.3, fy=0.3)
    cv2.namedWindow('img')
    cv2.setMouseCallback('img', input_mcqueen)

    mask = np.full_like(mcqueen, 255)
    points_mcqueen = np.float32([[0, 0], [mcqueen.shape[1], 0], [mcqueen.shape[1], mcqueen.shape[0]], [0, mcqueen.shape[0]]])

    while True:
        cv2.imshow('img', img)
        print(len(mcqueen_pos))
        if len(mcqueen_pos) == 4:
            # print('sa 4')
            mcqueen_pos = np.float32(mcqueen_pos)
            print('zamiana')
            transform = cv2.getPerspectiveTransform(points_mcqueen, mcqueen_pos)
            transformed_mcqueen = cv2.warpPerspective(mcqueen, transform, (img.shape[1], img.shape[0]))
            transformed_mcqueen_mask = cv2.warpPerspective(mask, transform, (img.shape[1], img.shape[0]))
            print('transformacje')
            connect = cv2.bitwise_and(img, cv2.bitwise_not(transformed_mcqueen_mask))
            merge = cv2.add(transformed_mcqueen, connect)
            print('laczenie')
            cv2.imshow('img', merge)

        if cv2.waitKey(200) == ord('q'):
            break

    cv2.destroyAllWindows()


def main():
    # todo_1()
    # todo_2()
    # todo_3()
    # todo_4()
    todo_5()


if __name__ == '__main__':
    main()
