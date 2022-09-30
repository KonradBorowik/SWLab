import cv2


def main():
    apexes = []
    clicked = 0
    def sample_callback(event, x, y, flag, param):
        if event == cv2.EVENT_RBUTTONDOWN:
            apexes.append([x, y])
            clicked = 1
            print(apexes)
            print(clicked)

    cv2.namedWindow("sample img")
    img = cv2.imread(r'pictures/lena.jpg')

    cv2.setMouseCallback('sample img', sample_callback)
    done = False
    while True:
        if not done and len(apexes) == 2:
            cv2.rectangle(img, apexes[0], apexes[1], (255, 0, 0), 5)
            mask = img[apexes[0][1]:apexes[1][1], apexes[0][0]:apexes[1][0]]

            canny = cv2.Canny(mask, 50, 150)
            canny = cv2.merge([canny, canny, canny])

            img[apexes[0][1]:apexes[1][1], apexes[0][0]:apexes[1][0]] = canny

            apexes = []

        cv2.imshow('sample img', img)
        if cv2.waitKey(50) == ord('q'):
            break


if __name__ == "__main__":
    main()
