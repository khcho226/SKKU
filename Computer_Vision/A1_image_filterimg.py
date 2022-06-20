import cv2
import numpy as np
import time

def get_gaussian_filter_1d(size, sigma):
    a = (size - 1) / 2
    b = np.array(range(int(-a), int(a) + 1))
    kernel_1d = np.exp(-(b * b) / (2 * sigma * sigma)) / (sigma * np.sqrt(2 * np.pi))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel = kernel_1d
    kernel = kernel.reshape((size, 1))
    return kernel

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

def cross_correlation_1d(img, kernel):
    a = img.shape[1]
    b = img.shape[0]
    filtered_img0 = np.full((b, a), 0, dtype = 'f8')
    ka = kernel.shape[1]
    kb = kernel.shape[0]
    img = padding(img, (kb, ka))
    for x in range(0, a):
        for y in range(0, b):
            filtered_img0[y, x] = np.sum(np.multiply(img[y:y + kb, x:x + ka], kernel))
    kernel1 = np.transpose(kernel)
    filtered_img = np.full((b, a), 0, dtype ='f8')
    ka = kernel1.shape[1]
    kb = kernel1.shape[0]
    img = padding(filtered_img0, (kb, ka))
    for x in range(0, a):
        for y in range(0, b):
            filtered_img[y, x] = np.sum(np.multiply(img[y:y + kb, x:x + ka], kernel1))
    return filtered_img

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

img = cv2.imread('lenna.png', cv2.IMREAD_GRAYSCALE)

g1 = get_gaussian_filter_1d(5, 1)
g2 = get_gaussian_filter_2d(5, 1)

print('get_gaussian_filter_1d(5, 1) : ', np.transpose(g1))
print('get_gaussian_filter_2d(5, 1) : ')
print(g2)

g21 = get_gaussian_filter_2d(5, 1)
g22 = get_gaussian_filter_2d(5, 6)
g23 = get_gaussian_filter_2d(5, 11)
g24 = get_gaussian_filter_2d(11, 1)
g25 = get_gaussian_filter_2d(11, 6)
g26 = get_gaussian_filter_2d(11, 11)
g27 = get_gaussian_filter_2d(17, 1)
g28 = get_gaussian_filter_2d(17, 6)
g29 = get_gaussian_filter_2d(17, 11)

img21 = cross_correlation_2d(img, g21)
cv2.putText(img21, '5x5 s=1', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img22 = cross_correlation_2d(img, g22)
cv2.putText(img22, '5x5 s=6', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img23 = cross_correlation_2d(img, g23)
cv2.putText(img23, '5x5 s=11', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img24 = cross_correlation_2d(img, g24)
cv2.putText(img24, '11x11 s=1', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img25 = cross_correlation_2d(img, g25)
cv2.putText(img25, '11x11 s=6', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img26 = cross_correlation_2d(img, g26)
cv2.putText(img26, '11x11 s=11', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img27 = cross_correlation_2d(img, g27)
cv2.putText(img27, '17x17 s=1', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img28 = cross_correlation_2d(img, g28)
cv2.putText(img28, '17x17 s=6', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img29 = cross_correlation_2d(img, g29)
cv2.putText(img29, '17x17 s=11', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

img101 = np.hstack([img21, img22, img23])
img102 = np.hstack([img24, img25, img26])
img103 = np.hstack([img27, img28, img29])
img104 = np.vstack([img101, img102, img103])
g3 = get_gaussian_filter_1d(17, 6)
g4 = get_gaussian_filter_2d(17, 6)

print('- lenna')
s1 = time.time()
e = cross_correlation_1d(img, g3)
print('1D filtering time : ', time.time() - s1)
s2 = time.time()
f = cross_correlation_2d(img, g4)
print('2D filtering time : ', time.time() - s2)
img1000 = e - f
cv2.imshow('pixel-wise difference map_lenna', img1000.astype('uint8'))

a10 = img1000.shape[1]
b10 = img1000.shape[0]
sum1 = 0
for x in range(0, a10):
    for y in range(0, b10):
        sum1 = sum1 + np.sqrt(img1000[x, y] * img1000[x, y])
print('Sum : ', sum1)
cv2.imwrite('result/part_1_gaussian_filtered_lenna.png', img104)
cv2.imshow('part_1_gaussian_filtered_lenna', img104.astype('uint8'))

print('- shapes')
img00 = cv2.imread('shapes.png', cv2.IMREAD_GRAYSCALE)

img21 = cross_correlation_2d(img00, g21)
cv2.putText(img21, '5x5 s=1', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img22 = cross_correlation_2d(img00, g22)
cv2.putText(img22, '5x5 s=6', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img23 = cross_correlation_2d(img00, g23)
cv2.putText(img23, '5x5 s=11', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img24 = cross_correlation_2d(img00, g24)
cv2.putText(img24, '11x11 s=1', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img25 = cross_correlation_2d(img00, g25)
cv2.putText(img25, '11x11 s=6', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img26 = cross_correlation_2d(img00, g26)
cv2.putText(img26, '11x11 s=11', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img27 = cross_correlation_2d(img00, g27)
cv2.putText(img27, '17x17 s=1', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img28 = cross_correlation_2d(img00, g28)
cv2.putText(img28, '17x17 s=6', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
img29 = cross_correlation_2d(img00, g29)
cv2.putText(img29, '17x17 s=11', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)

img101 = np.hstack([img21, img22, img23])
img102 = np.hstack([img24, img25, img26])
img103 = np.hstack([img27, img28, img29])
img104 = np.vstack([img101, img102, img103])

s1 = time.time()
e = cross_correlation_1d(img00, g3)
print('1D filtering time : ', time.time() - s1)
s2 = time.time()
f = cross_correlation_2d(img00, g4)
print('2D filtering time : ', time.time() - s2)
img1000 = e - f
cv2.imshow('pixel-wise difference map_shapes', img1000.astype('uint8'))

a10 = img1000.shape[1]
b10 = img1000.shape[0]
sum1 = 0
for x in range(0, a10):
    for y in range(0, b10):
        sum1 = sum1 + np.sqrt(img1000[y, x] * img1000[y, x])
print('Sum : ', sum1)
cv2.imwrite('result/part_1_gaussian_filtered_shapes.png', img104)
cv2.imshow('part_1_gaussian_filtered_shapes', img104.astype('uint8'))

cv2.waitKey(0)
cv2.destroyAllWindows()