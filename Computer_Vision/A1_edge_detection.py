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

def compute_image_gradiant(img):
    a = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    b = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    a1 = cross_correlation_2d(img, a)
    b1 = cross_correlation_2d(img, b)
    mag = np.sqrt((a1 * a1) + (b1 * b1))
    ang1 = np.arctan2(*(1, 0)[::-1])
    ang2 = np.arctan2(*(a1, b1)[::-1])
    dir = np.rad2deg((ang1 - ang2) % (2 * np.pi))
    dir = (360 - dir) % 360
    return mag, dir

def non_maximum_suppression_dir(mag, dir):
    a = mag.shape[1]
    b = mag.shape[0]
    suppressed_mag = mag
    for x in range(1, a - 1):
        for y in range(1, b - 1):
            if 22.5 <= dir[y, x] < 67.5:
                if mag[y, x] < mag[y + 1, x - 1] or mag[y, x] < mag[y - 1, x + 1]:
                    suppressed_mag[y, x] = 0
                else:
                    suppressed_mag[y, x] = mag[y, x]
            elif 67.5 <= dir[y, x] < 112.5:
                if mag[y, x] < mag[y - 1, x] or mag[y, x] < mag[y + 1, x]:
                    suppressed_mag[y, x] = 0
                else:
                    suppressed_mag[y, x] = mag[y, x]
            elif 112.5 <= dir[y, x] < 157.5:
                if mag[y, x] < mag[y - 1, x - 1] or mag[y, x] < mag[y + 1, x + 1]:
                    suppressed_mag[y, x] = 0
                else:
                    suppressed_mag[y, x] = mag[y, x]
            elif 157.5 <= dir[y, x] < 202.5:
                if mag[y, x] < mag[y, x - 1] or mag[y, x] < mag[y, x + 1]:
                    suppressed_mag[y, x] = 0
                else:
                    suppressed_mag[y, x] = mag[y, x]
            elif 202.5 <= dir[y, x] < 247.5:
                if mag[y, x] < mag[y + 1, x - 1] or mag[y, x] < mag[y - 1, x + 1]:
                    suppressed_mag[y, x] = 0
                else:
                    suppressed_mag[y, x] = mag[y, x]
            elif 247.5 <= dir[y, x] < 292.5:
                if mag[y, x] < mag[y - 1, x] or mag[y, x] < mag[y + 1, x]:
                    suppressed_mag[y, x] = 0
                else:
                    suppressed_mag[y, x] = mag[y, x]
            elif 292.5 <= dir[y, x] < 337.5:
                if mag[y, x] < mag[y - 1, x - 1] or mag[y, x] < mag[y + 1, x + 1]:
                    suppressed_mag[y, x] = 0
                else:
                    suppressed_mag[y, x] = mag[y, x]
            elif 337.5 <= dir[y, x] <= 360 or 0 <= dir[y, x] < 22.5:
                if mag[y, x] < mag[y, x - 1] or mag[y, x] < mag[y, x + 1]:
                    suppressed_mag[y, x] = 0
                else:
                    suppressed_mag[y, x] = mag[y, x]
    return suppressed_mag

img1 = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)
g = get_gaussian_filter_2d(7, 1.5)
img1 = cross_correlation_2d(img1, g)
img2 = cross_correlation_2d(img2, g)

print('- lenna')
s1 = time.time()
mag, dir = compute_image_gradiant(img1)
print('compute_image_gradiant time : ', time.time() - s1)
img1 = cv2.convertScaleAbs(mag)
cv2.imshow('part_2_edge_raw_lenna', img1.astype('uint8'))
cv2.imwrite('result/part_2_edge_raw_lenna.png', img1)

s11 = time.time()
mag = non_maximum_suppression_dir(mag, dir)
print('non_maximum_suppression_dir time : ', time.time() - s11)
img1 = cv2.convertScaleAbs(mag)
cv2.imshow('part_2_edge_sup_lenna', img1.astype('uint8'))
cv2.imwrite('result/part_2_edge_sup_lenna.png', img1)

print('- shapes')
s2 = time.time()
mag, dir = compute_image_gradiant(img2)
print('compute_image_gradiant time : ', time.time() - s2)
img2 = cv2.convertScaleAbs(mag)
cv2.imshow('part_2_edge_raw_shapes', img2.astype('uint8'))
cv2.imwrite('result/part_2_edge_raw_shapes.png', img2)

s22 = time.time()
mag = non_maximum_suppression_dir(mag, dir)
print('non_maximum_suppression_dir time : ', time.time() - s22)
img2 = cv2.convertScaleAbs(mag)
cv2.imshow('part_2_edge_sup_shapes', img2.astype('uint8'))
cv2.imwrite('result/part_2_edge_sup_shapes.png', img2)

cv2.waitKey(0)
cv2.destroyAllWindows()