import cv2
import numpy as np
import time

def get_gaussian_filter_2d(size, sigma):
    a = (size - 1) / 2
    b = np.array(range(int(-a), int(a) + 1))
    kernel_2d = np.exp(-(b * b) / (2 * sigma * sigma)) / (sigma * np.sqrt(2 * np.pi))
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel_2d = np.outer(kernel_2d, kernel_2d)
    kernel_2d = kernel_2d / kernel_2d.sum()
    kernel = kernel_2d
    return kernel

def padding(img, kernel_size):
    a = img.shape[1]
    b = img.shape[0]
    a1 = int((kernel_size[1] - 1) / 2)
    b1 = int((kernel_size[0] - 1) / 2)
    pad = np.full((b + (2 * b1), a + (2 * a1)), 0, dtype = 'f8')
    if a1 == 0:
        pad[b1:-b1, 0:] = img.copy()
    elif b1 == 0:
        pad[0:, a1:-a1] = img.copy()
    else:
        pad[b1:-b1, a1:-a1] = img.copy()
    pad[0:b1, 0:a1] = img[0, 0]
    pad[0:b1, -a1:] = img[0, a - 1]
    pad[-b1:, 0:a1] = img[b - 1, 0]
    pad[-b1:, -a1:] = img[b - 1, a - 1]
    if a1 == 0:
        for x in range(a1, a1 + a):
            pad[0:b1, x] = img[0, x - a1]
        for x in range(a1, a1 + a):
            pad[-b1:, x] = img[b - 1, x - a1]
    elif b1 == 0:
        for x in range(b1, b1 + b):
            pad[x, 0:a1] = img[x - b1, 0]
        for x in range(b1, b1 + b):
            pad[x, -a1:] = img[x - b1, a - 1]
    else:
        for x in range(a1, a1 + a):
            pad[0:b1, x] = img[0, x - a1]
        for x in range(a1, a1 + a):
            pad[-b1:, x] = img[b - 1, x - a1]
        for x in range(b1, b1 + b):
            pad[x, 0:a1] = img[x - b1, 0]
        for x in range(b1, b1 + b):
            pad[x, -a1:] = img[x - b1, a - 1]
    return pad

def cross_correlation_2d(img, kernel):
    a = img.shape[1]
    b = img.shape[0]
    filtered_img = np.full((b, a), 0, dtype = 'f8')
    ka = kernel.shape[1]
    kb = kernel.shape[0]
    img = padding(img, (kb, ka))
    for x in range(0, a):
        for y in range(0, b):
            filtered_img[y, x] = np.sum(np.multiply(img[y:y + kb, x:x + ka], kernel))
    return filtered_img

def compute_corner_response(img):
    a = img.shape[1]
    b = img.shape[0]
    c = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    d = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    sx = cross_correlation_2d(img, c)
    sy = cross_correlation_2d(img, d)
    R = np.full((b, a), 0, dtype = 'f8')
    xx = sx * sx
    xy = sx * sy
    yy = sy * sy
    for x in range(2, a - 2):
        for y in range(2, b - 2):
            xxxx = (xx[y - 2:y + 2, x - 2:x + 2]).sum()
            xxyy = (xy[y - 2:y + 2, x - 2:x + 2]).sum()
            yyyy = (yy[y - 2:y + 2, x - 2:x + 2]).sum()
            R[y, x] = ((xxxx * yyyy) - (xxyy * xxyy)) - (0.04 * (xxxx + yyyy) * (xxxx + yyyy))
            if R[y, x] < 0:
                R[y, x] = 0
    cv2.normalize(R, R, 1.0, 0.0, cv2.NORM_MINMAX)
    return R

def non_maximum_suppression_win(R, winSize):
    a = R.shape[1]
    b = R.shape[0]
    s = int((winSize - 1) / 2)
    suppressed_R = R
    for x in range(s, a - s):
        for y in range(s, b - s):
            for z in range(x - s, x + s):
                for w in range(y - s, y + s):
                    if R[y, x] < R[w, z]:
                        suppressed_R[y, x] = 0
            if R[y, x] < 0.1:
                suppressed_R[y, x] = 0
    return suppressed_R

img1 = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)
g = get_gaussian_filter_2d(7, 1.5)
img1 = cross_correlation_2d(img1, g)
img2 = cross_correlation_2d(img2, g)

print('- lenna')
t1 = time.time()
R1 = compute_corner_response(img1)
print('compute_corner_response time : ', time.time() - t1)
cv2.imshow('part_3_corner_raw_lenna', R1)
cv2.imwrite('result/part_3_corner_raw_lenna.png', R1 * 255)

cv2.imwrite('lenna_1.png', img1)
img11 = cv2.imread('lenna_1.png')
img111 = cv2.imread('lenna_1.png')
a1 = R1.shape[1]
b1 = R1.shape[0]
for x in range(0, a1 - 1):
    for y in range(0, b1 - 1):
        if R1[y, x] > 0.1:
            img11[y, x] = [0, 255, 0]
cv2.imshow('part_3_corner_bin_lenna', img11)
cv2.imwrite('result/part_3_corner_bin_lenna.png', img11)

t11 = time.time()
R11 = non_maximum_suppression_win(R1, 11)
print('non-maximum_suppression time : ', time.time() - t11)
for x in range(0, a1 - 1):
    for y in range(0, b1 - 1):
        if R11[y, x] > 0.1:
            img111 = cv2.circle(img111, (x, y), 5, (0, 255, 0), 2)
cv2.imshow('part_3_corner_sup_lenna', img111)
cv2.imwrite('result/part_3_corner_sup_lenna.png', img111)

print('- shapes')
t2 = time.time()
R2 = compute_corner_response(img2)
print('compute_corner_response time : ', time.time() - t2)
cv2.imshow('part_3_corner_raw_shapes', R2)
cv2.imwrite('result/part_3_corner_raw_shapes.png', R2 * 255)

cv2.imwrite('shapes_1.png', img2)
img22 = cv2.imread('shapes_1.png')
img222 = cv2.imread('shapes_1.png')
a1 = R2.shape[1]
b1 = R2.shape[0]
for x in range(0, a1 - 1):
    for y in range(0, b1 - 1):
        if R2[y, x] > 0.1:
            img22[y, x] = [0, 255, 0]
cv2.imshow('part_3_corner_bin_shapes', img22)
cv2.imwrite('result/part_3_corner_bin_shapes.png', img22)

t22 = time.time()
R22 = non_maximum_suppression_win(R2, 11)
print('non-maximum_suppression time : ', time.time() - t22)
for x in range(0, a1 - 1):
    for y in range(0, b1 - 1):
        if R22[y, x] > 0.1:
            img222 = cv2.circle(img222, (x, y), 5, (0, 255, 0), 2)
cv2.imshow('part_3_corner_sup_shapes', img222)
cv2.imwrite('result/part_3_corner_sup_shapes.png', img222)

cv2.waitKey(0)
cv2.destroyAllWindows()