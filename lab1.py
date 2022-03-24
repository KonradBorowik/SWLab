import cv2
from matplotlib import pyplot as plt


def ex_1():
    cap = cv2.VideoCapture(0)  # open the default camera

    key = ord('a')
    while key != ord('q'):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame comes here
        # Convert RGB image to grayscale
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Blur the image
        img_filtered = cv2.GaussianBlur(img_gray, (7, 7), 1.5)
        # Detect edges on the blurred image
        img_edges = cv2.Canny(img_filtered, 0, 30, 3)

        # Display the result of our processing
        cv2.imshow('result', img_edges)
        # Wait a little (30 ms) for a key press - this is required to refresh the image in our window
        key = cv2.waitKey(30)

    # When everything done, release the capture
    cap.release()
    # and destroy created windows, so that they are not left for the rest of the program
    cv2.destroyAllWindows()


def ex_2():
    pic = cv2.imread(r"C:\Users\konra\Pictures\lena.jpg")

    cv2.imshow("obrazek", pic)
    cv2.waitKey(0)

    cv2.imwrite(r"C:\Users\konra\Pictures\moja_lena.jpg", pic)
    my_pic = cv2.imread(r"C:\Users\konra\Pictures\moja_lena.jpg")
    cv2.imshow("mój obrazek", my_pic)
    cv2.waitKey(0)


def ex_3():
    pic = cv2.imread(r"C:\Users\konra\Pictures\lena.jpg")
    print(pic.shape)
    px = pic[220,270]
    print(f'Pixel value at [220, 270]: {px}')

    gray_pic = cv2.cvtColor(pic, cv2.COLOR_BGR2GRAY)
    gray_px = gray_pic[220,270]
    cv2.imshow("szary", gray_pic)
    cv2.waitKey(0)
    print(f'Luminance value at [220, 270]: {gray_px}')


def ex_4():
    pic = cv2.imread(r"C:\Users\konra\Pictures\lena.jpg")
    hat = pic[0:250, 100:400]

    pic[250:500, 100:400] = hat

    cv2.imshow("aaa", pic)
    cv2.waitKey(0)


def ex_5():
    pic = cv2.imread(r"C:\Users\konra\Pictures\lena.jpg")
    cv2.imshow("lena", pic)
    cv2.waitKey(0)

    plt.imshow(pic)
    plt.show()

# def ex_6():
#     na tablicy, wiem o co biega


def hw_3():
    pic1 = cv2.imread("pictures\kubus.jpg")
    pic2 = cv2.imread("pictures\Winnie_Pooh.jpg")
    pic2 = cv2.resize(pic2, None, fx=0.3, fy=0.3)
    pic3 = cv2.imread("pictures\puchatek.jpg")

    pics = [pic1, pic2, pic3]
    i = 0
    while True:
        cv2.destroyAllWindows()
        cv2.imshow(f"picture {i}", pics[i])
        key = cv2.waitKey(10000)

        if key == ord('q'):
            i = i - 1
            if i < 0:
                i = 2
        elif key == ord('w'):
            i = i + 1
            if i > 2:
                i = 0
        elif key == ord('b'):
            break
        print(i)


def main():
    # ex_1()
    # ex_2()
    # ex_3()
    # ex_4()
    # ex_5()
    # ex_6()
    # ex_7()
    hw_3()


if __name__ == '__main__':
   main()
