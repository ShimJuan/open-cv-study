import cv2
import numpy as np

def color_op():
    src = cv2.imread('butterfly.jpg', cv2.IMREAD_COLOR)

    if src is None:
        print('image load failed')
        return
    
    print('src.shape:',src.shape)
    print('src.dtype:',src.dtype)
    # b,g,r = src[0,0]
    print('The pixel value [B,G,R] at (0,0) is',src[0,0])

def color_inverse():
    src = cv2.imread('butterfly.jpg')

    if src is None:
        print('image load failed')
        return
    dst = np.zeros(src.shape, src.dtype)

    for j in range(src.shape[0]):
        for i in range(src.shape[1]):
            p1 = src[j,i]
            p2 = dst[j,i]

            p2[0] = 255 - p1[0]
            p2[1] = 255 - p1[1]
            p2[2] = 255 - p1[2]
    cv2.imshow('src', src)
    cv2.imshow('dst', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
#color_inverse()
##
def color_split():
    src = cv2.imread('candies.png',cv2.IMREAD_COLOR)

    if src is None:
        print('image load failed')
        return
    # b_plane, g_plane, r_plane = cv2.split(src)
    bgr_planes = cv2.split(src)

    cv2.imshow('src',src)
    cv2.imshow('B_plane',bgr_planes[0])
    cv2.imshow('G_plane',bgr_planes[1])
    cv2.imshow('R_plane',bgr_planes[2])
    cv2.waitKey()
    cv2.destroyAllWindows()
# color_split()
#######################

def on_hue_changed(_=None):
    lower_hue = cv2.getTrackbarPos('Lower Hue', 'mask')
    upper_hue = cv2.getTrackbarPos('Upper Hue', 'mask')

    lowerb = (lower_hue, 100, 0)
    upperb = (upper_hue, 255, 255)
    mask = cv2.inRange(src_hsv, lowerb, upperb)

    cv2.imshow('mask',mask)

def main():
    global src_hsv

    src = cv2.imread('candies.png',cv2.IMREAD_COLOR)

    if src is None:
        print('image load failed')
        return
    
    src_hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

    cv2.imshow('src',src)

    cv2.namedWindow('mask')
    cv2.createTrackbar('Lower Hue','mask',40, 179, on_hue_changed)
    cv2.createTrackbar('Upper Hue','mask',80, 179, on_hue_changed)
    on_hue_changed(0)

    cv2.waitKey()
    cv2.destroyAllWindows()

ref = cv2.imread('ref.png',cv2.IMREAD_COLOR)
mask = cv2.imread('mask.bmp',cv2.IMREAD_GRAYSCALE)
ref_ycrcb = cv2.cvtColor(ref,cv2.COLOR_BGR2YCrCb)

channels = [1,2]
cr_bins = 128
cb_bins = 128
histSize = [cr_bins, cb_bins]
cr_range = [0, 256]
cb_range = [0, 256]
ranges = cr_range + cb_range

hist = cv2.calcHist([ref_ycrcb],channels, mask, histSize, ranges)

# Apply histogram backprojection to an input image

src = cv2.imread('kids.png',cv2.IMREAD_COLOR)
src_ycrcb = cv2.cvtColor(src, cv2.COLOR_BGR2YCrCb)

backproj = cv2.calcBackProject([src_ycrcb], channels, hist, ranges, 1)
cv2.imshow('src',src)
cv2.imshow('backproj',backproj)
cv2.waitKey()
cv2.destroyAllWindows()
# main()