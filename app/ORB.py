import cv2
import numpy as np

# initialize ORB object
orb_detector = cv2.ORB_create()


def detectFeatures():

    img_1 = cv2.imread('img1.png')
    img_2 = cv2.imread('img2.png')

    grayscale_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
    grayscale_2 = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)

    img1LocationPoints, img1QueryPoints = orb_detector.detectAndCompute(
        grayscale_1, None)
    img2LocationPoints, img2QueryPoints = orb_detector.detectAndCompute(
        grayscale_2, None)

    matcher = cv2.BFMatcher()
    matches = matcher.match(img1QueryPoints, img2QueryPoints)

    matched_image = cv2.drawMatches(
        img_1, img1LocationPoints, img_2, img2LocationPoints, matches[:15], None)

    matched_image = cv2.resize(matched_image, (1000, 650))

    return matched_image
