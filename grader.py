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

# correct answer key
ANSWER_KEY = {0: 3, 1: 4, 2: 4, 3: 0, 4: 3}

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
edge = cv2.Canny(blur, 75, 200)

# initialize the contour that corresponds to the document
cnts = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None

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
# obtain a top-down birds eye view of the paper
paper = four_point_transform(image, docCnt.reshape(4, 2))
gray_paper = four_point_transform(gray, docCnt.reshape(4, 2))

thresh = cv2.threshold(gray_paper, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# initialize list of contours that correspond to questions
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

for cn in cnts:
    # compute the bounding box of the contour, then use the
    # bounding box to derive the aspect(width to height) ratio
    (x, y, w, h) = cv2.boundingRect(cn)
    ar = w / float(h)

    # in order to label the contour as a question, region
    # should be sufficiently wide and tall
    # aspect ratio approximately equal to 1
    if w >= 20 and h >= 20 and ar >= 0.9 and ar <= 1.1:
        questionCnts.append(cn)

# GRADING PART #

# sort the question contours top to bottom
# initialize total number of correct answers
questionCnts = contours.sort_contours(questionCnts,
                                      method="top-to-bottom")[0]
correct = 0

for (q, i) in enumerate(np.arange(0, len(questionCnts), 5)):
    # sort the contours for the current question from
    # left to right, then initialize the index of the
    # bubbled answer
    cnts = contours.sort_contours(questionCnts[i:i + 5])[0]
    bubbled = None

# To know which bubble is filled in:

# loop over sorted contours
for (j, c) in enumerate(cnts):
    # mask which reveals only current bubble for quest
    mask = np.zeros(thresh.shape, dtype="uint8")
    cv2.drawContours(mask, [c], -1, 255, -1)

    # apply the mask to the thresholded image, then
    # count the number of non-zero pixels in the
    # bubble area
    mask = cv2.bitwise_and(thresh, thresh, mask=mask)
    total = cv2.countNonZero(mask)

    # if the current total has a larger number of total
    # non-zero pixels, then we are examining the currently
    # bubbled-in answer
    if bubbled is None or total > bubbled[0]:
        bubbled = (total, j)

# Initialize the correct answer
color = (0, 0, 255)
k = ANSWER_KEY[q]

if k == bubbled[1]:
    color = (0, 255, 0)
    correct += 1

cv2.drawContours(paper, [cnts[k]], -1, color, 3)

score = (correct / 5.0) * 100
print("[INFO] Score: {: .2f}%".format(score))
cv2.putText(paper, "{:.2f}%".format(score), (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
cv2.imshow("Original", image)
cv2.imshow("Exam", paper)
cv2.waitKey(0)
