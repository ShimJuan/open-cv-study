import cv2
import numpy as np
import sys

def on_threshold(pos):
    _, dst = cv2.threshold(src, pos, 255, cv2.THRESH_BINARY)
    cv2.imshow('dst',dst)

filename = 'neutrophils.png'
if len(sys.argv) > 1:
    filename = sys.argv[1]
src = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)

if src is None:
    print('image load failed')
    sys.exit()
cv2.imshow('src',src)
cv2.namedWindow('dst')
cv2.createTrackbar('Threshold','dst',0, 255, on_threshold)
cv2.setTrackbarPos('Threshold','dst',128)
cv2.waitKey()
cv2.destroyAllWindows()

#on_threshold()
##################################3
def on_trackbar(pos):
    bsize = pos
    if bsize % 2 == 0: bsize = bsize -1
    if bsize < 3: bsize = 3

    dst = cv2.adaptiveThreshold(src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,bsize,5)

    cv2.imshow('dst',dst)

src = cv2.imread('sudoku.jpg', cv2.IMREAD_GRAYSCALE)
if src is None:
    print('image load failed')
    sys.exit()

cv2.imshow('src',src)
cv2.namedWindow('dst')
cv2.createTrackbar('Block Size', 'dst', 0, 200, on_trackbar)
cv2.setTrackbarPos('Block Size', 'dst', 11)
cv2.waitKey()
cv2.destroyAllWindows()
on_trackbar()
