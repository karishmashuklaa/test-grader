from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
                help="Path to the test image")
args = vars(ap.parse_args())

# defining the correct answer key
ANSWER_KEY = {0: 3, 1: 4, 2: 4, 3: 0, 4: 3}

# load and convert image to grayscale, blur it
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edge = cv2.Canny(blur, 75, 200)

# cv2.imshow("Test Grader", edge)
# cv2.waitKey(0)

# find contours in the edge map, then initialize
# the contour that corresponds to the document
cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

# ensure at least one contour is found
if len(cnts) > 0:
    # sort contours in desc order based on their area
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    for cn in cnts:
        peri = cv2.arcLength(cn, True)
        approx = cv2.approxPolyDP(cn, 0.02 * peri, True)

        # if four points found, then we have found the test sheet
        if len(approx) == 4:
            docCnt = approx
            break

# apply a four point perspective transform to
# original and grayscale image to obtain a top-down
# birds eye view of the paper
paper = four_point_transform(image, docCnt.reshape(4, 2))
gray_img = four_point_transform(gray, docCnt.reshape(4, 2))

