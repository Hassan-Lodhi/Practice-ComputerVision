import cv2 as cv
import numpy as np

# using imread function
# blank = np.zeros((500, 500, 3), dtype='int8')
# cv.imshow('Blank', blank)

img = cv.imread('images/cat.jpg')
cv.imshow('cat', img)

# 1-Paint the images
# blank[:] = 0,255,0
# blank[:] = 255,2,1
# cv.imshow('grey', blank)

# 2-Draw a rectangle
# cv.rectangle(blank, (0,0), (250, 250), (0,255,0), thickness=2)
# cv.imshow('Rectangle', blank)
# cv.waitKey(0)

# draw line
# cv.line(blank, (100,100), (300,400),(255,255,255), thickness=3)
# cv.imshow('line', blank)

# 3-Putting text in it
# cv.putText(blank, 'Hello Hassan ', (255, 255), cv.FONT_HERSHEY_TRIPLEX, 1.0, (0, 255, 0), 2)
# cv.imshow('Text', blank)

# Converting to grey Scale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)
# cv.waitKey(0)

# Blurring the image
# blur = cv.GaussianBlur(img, (7, 7), cv.BORDER_DEFAULT)
# cv.imshow('Blur', blur)

# Edge cascade
canny = cv.Canny(img, 125, 125)
cv.imshow('Canny Edges', canny)

# casacading with bluree images
# canny = cv.Canny(blur, 125, 125)
# cv.imshow('Canny Edges', canny)

# Dilating cv2.dilate()
dilated = cv.dilate(canny, (7, 7), iterations=1)
cv.imshow('Dilated', dilated)

# Eroding
eroded = cv.erode(dilated, (7, 7), iterations=1)
cv.imshow('Eroded', eroded)

# Resize
resized = cv.resize(img, (500, 500), interpolation=cv.INTER_CUBIC)
cv.imshow('Resized', resized)

# Crapping
croped = img[50:220, 210:410]
cv.imshow('Cropped', croped)

# Simple Thresholding
threshold, thresh = cv.threshold(gray, 150, 255, cv.THRESH_BINARY)
cv.imshow('Threshold', thresh)

# Inverse Thresholding
# threshold, thresh_inv = cv.threshold(gray, 150, 255, cv.THRESH_BINARY_INV)
# cv.imshow('Threshold Inverse', thresh_inv)

# Adaptive Thresholding
adaptive_thresh = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 13, 3)
cv.imshow('Threshold Adaptive', adaptive_thresh)

cv.waitKey(0)
