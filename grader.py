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

cv2.imshow("Test Grader", edge)
cv2.waitKey(0)

