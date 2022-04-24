import cv2
import numpy as np

def task_1():
    forward1 = cv2.imread(r'pictures\lab7\forward-1.bmp')
    forward2 = cv2.imread(r'pictures\lab7\forward-2.bmp')

    fast : cv2.FastFeatureDetector = cv2.FastFeatureDetector_create()
    fast_keypoints = fast.detect(forward1, None)
    fast_forward1 = cv2.drawKeypoints(forward1, fast_keypoints, None, (0, 0, 255))

    orb = cv2.ORB_create()
    orb_keypoints = orb.detect(forward1, None)
    orb_forward1 = cv2.drawKeypoints(forward1, orb_keypoints, None, (0, 0, 255))

    print(cv2.DescriptorMatcher.match([orb, fast]))

    cv2.imshow('fast', fast_forward1)
    cv2.imshow('orb', orb_forward1)
    cv2.waitKey()


def task_2():
    perspective1 = cv2.imread(r'pictures\lab7\perspective-1.bmp', cv2.IMREAD_GRAYSCALE)
    perspective2 = cv2.imread(r'pictures\lab7\perspective-3.bmp', cv2.IMREAD_GRAYSCALE)
    # ORB
    orb : cv2.ORB = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(perspective1, None)
    kp2, des2 = orb.detectAndCompute(perspective2, None)

    # create brute-force matcher
    bf : cv2.BFMatcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, True)

    matches = bf.match(des1, des2)

    # sort by distance
    matches = sorted(matches, key=lambda x: x.distance)
    # draw first 10 matches
    match = cv2.drawMatches(perspective1, kp1, perspective2, kp2, matches[:10], outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('match', match)
    cv2.waitKey()


def task_2_1():
    perspective1 = cv2.imread(r'pictures\lab7\perspective-1.bmp', cv2.IMREAD_GRAYSCALE)
    perspective2 = cv2.imread(r'pictures\lab7\perspective-3.bmp', cv2.IMREAD_GRAYSCALE)

    # Initiate FAST detector
    star = cv2.xfeatures2d.StarDetector_create()
    # detect keypoints
    kp1 = star.detect(perspective1, None)
    kp2 = star.detect(perspective2, None)

    # BRIEF
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp1, des1 = brief.compute(perspective1, kp1)
    kp2, des2 = brief.compute(perspective2, kp2)
    # matcher
    bf : cv2.BFMatcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)
    match = cv2.drawMatches(perspective1, kp1, perspective2, kp2, matches[:10], outImg=None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('b_match', match)
    cv2.waitKey()


def main():
    # task_1()
    # task_2()
    task_2_1()
    # ta sk_3()


if __name__ == "__main__":
    main()