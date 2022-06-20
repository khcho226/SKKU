import cv2
import numpy as np
import os
import math

sift_dir = os.listdir('./sift')
type = np.dtype([('val1', '<B')])
count = 0
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
flags = cv2.KMEANS_RANDOM_CENTERS
list = []

while True :
    order = 0
    data = np.fromfile('./sift/' + sift_dir[count], type)
    num = len(data['val1']) / 128

    for x in range(0, int(num)) :
        list_0 = []
        for y in range(0, 128) :
            list_0.append(data['val1'][order])
            order = order + 1
        list.append(list_0)

    count = count + 1
    if count == 1000:
        break

K = 1000
a, b, c = cv2.kmeans(np.float32(list), K, None, criteria, 10, flags)
count = 0
f = open('final.txt', 'w')
f.close()
total = []
c = np.array(c)
np.save('array', c)
c = np.load('array.npy')
print(c)
print('start')

while True :
    order = 0
    data = np.fromfile('./sift/' + sift_dir[count], type)
    num = len(data['val1']) / 128
    his = []
    for x in range(K):
        his.append(0)

    for x in range(0, int(num)) :
        list_0 = []
        for y in range(0, 128) :
            list_0.append(data['val1'][order])
            order = order + 1

        min = 10000000
        min_num = 0
        for y in range(0, K) :
            sum = 0
            for z in range(0, 128) :
                sum = sum + (c[y][z] - list_0[z]) * (c[y][z] - list_0[z])
            sum = math.sqrt(sum)
            if sum < min :
                min = sum
                min_num = y
        his[min_num] = his[min_num] + 1

    for x in range(K) :
        his[x] = float(his[x] / int(num))
    total.append(his)

    print(count)
    count = count + 1
    if count == 1000 :
        break

np.savetxt('final.txt', total)
dim = np.array([1000, K])
text = np.loadtxt('final.txt')
with open('A4_2016311209.des', 'wb') as f :
    dim.tofile(f, format = 'i')
    text.tofile(f, format = 'f')