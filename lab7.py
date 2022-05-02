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


def panorama():
    '''
    create panorama from two pictures
    '''

    # read images
    left = cv2.imread(r'pictures\left.jpg')
    left = cv2.resize(left, (0, 0), fx=0.6, fy=0.6)
    left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)

    right = cv2.imread(r'pictures\right.jpg')
    right = cv2.resize(right, (0, 0), fx=0.6, fy=0.6)
    right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # Initiate FAST detector
    star = cv2.xfeatures2d.StarDetector_create()
    # detect keypoints
    kp_l = star.detect(left_gray, None)
    kp_r = star.detect(right_gray, None)

    # BRIEF
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp_l, des_l = brief.compute(left_gray, kp_l)
    kp_r, des_r = brief.compute(right_gray, kp_r)

    kp1 = []
    for kp in range(len(kp_l)):
        kp1.append([int(kp_l[kp].pt[0]), int(kp_l[kp].pt[1])])

    kp2 = []
    for kp in range(len(kp_r)):
        kp2.append([int(kp_r[kp].pt[0]), int(kp_r[kp].pt[1])])

    # matching
    bf : cv2.BFMatcher = cv2.BFMatcher_create(cv2.NORM_HAMMING, crossCheck=True)

    matches = bf.match(des_r, des_l)

    coords_kp_l = []
    coords_kp_r = []
    for i in matches:
        id2 = i.queryIdx
        id1 = i.trainIdx

        (x, y) = kp1[id1]
        coords_kp_l.append((x, y))
        (x, y) = kp2[id2]
        coords_kp_r.append((x, y))

    coords_kp_r = np.array(coords_kp_r)
    coords_kp_l = np.array(coords_kp_l)

    # homography
    H, _ = cv2.findHomography(coords_kp_r, coords_kp_l, cv2.RANSAC, 5.0)

    # Apply panorama correction
    width = right.shape[1] + left.shape[1]
    height = right.shape[0] + left.shape[0]

    result = cv2.warpPerspective(right, H, (width, height))
    result[0:left.shape[0], 0:left.shape[1]] = left

    # drawing on pictures
    left_kp = cv2.drawKeypoints(left.copy(), kp_l, None, (0, 0, 255))
    right_kp = cv2.drawKeypoints(right.copy(), kp_r, None, (0, 0, 255))

    matched_keypoints = cv2.drawMatches(right.copy(), kp_r, left.copy(), kp_l, matches, None)

    # showing pictures
    cv2.imshow('left', left_kp)
    cv2.imshow('right', right_kp)
    cv2.imshow('matched keypoints', matched_keypoints)
    cv2.imshow('panorama', result)
    cv2.waitKey()


def main():
    # task_1()
    # task_2()
    # task_2_1()
    panorama()


if __name__ == "__main__":
    main()