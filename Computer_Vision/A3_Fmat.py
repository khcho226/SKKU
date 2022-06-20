import cv2
import numpy as np
import random
import keyboard
import time

def compute_avg_reproj_error(_M, _F):
    N = _M.shape[0]
    X = np.c_[ _M[:,0:2] , np.ones( (N,1) ) ].transpose()
    L = np.matmul( _F , X ).transpose()
    norms = np.sqrt( L[:,0]**2 + L[:,1]**2 )
    L = np.divide( L , np.kron( np.ones( (3,1) ) , norms ).transpose() )
    L = ( np.multiply( L , np.c_[ _M[:,2:4] , np.ones( (N,1) ) ] ) ).sum(axis=1)
    error = (np.fabs(L)).sum()
    X = np.c_[_M[:, 2:4], np.ones((N, 1))].transpose()
    L = np.matmul(_F.transpose(), X).transpose()
    norms = np.sqrt(L[:, 0] ** 2 + L[:, 1] ** 2)
    L = np.divide(L, np.kron(np.ones((3, 1)), norms).transpose())
    L = ( np.multiply( L , np.c_[ _M[:,0:2] , np.ones( (N,1) ) ] ) ).sum(axis=1)
    error += (np.fabs(L)).sum()
    return error/(N*2)

def compute_F_raw(M) :
    B = []
    for x in range(0, M.shape[0]) :
        x1 = M[x][0]
        y1 = M[x][1]
        x2 = M[x][2]
        y2 = M[x][3]
        A = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]
        B.append(A)
    B = np.array(B).astype(np.float64)
    U, D, V = np.linalg.svd(B, full_matrices = True)
    min = np.argmin(D)
    F = np.array([[V[min][0], V[min][1], V[min][2]], [V[min][3], V[min][4], V[min][5]], [V[min][6], V[min][7], V[min][8]]])
    return F

def compute_F_norm(M) :
    M2 = np.zeros((M.shape[0], M.shape[1]))
    for x in range(0, M.shape[0]) :
        M2[x][0] = (M[x][0] - row) * row_m
        M2[x][1] = (M[x][1] - col) * col_m
        M2[x][2] = (M[x][2] - row) * row_m
        M2[x][3] = (M[x][3] - col) * col_m
    B = []
    for x in range(0, M.shape[0]) :
        x1 = M2[x][0]
        y1 = M2[x][1]
        x2 = M2[x][2]
        y2 = M2[x][3]
        A = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]
        B.append(A)
    B = np.array(B).astype(np.float64)
    U, D, V = np.linalg.svd(B, full_matrices = True)
    min = np.argmin(D)
    F = np.array([[V[min][0], V[min][1], V[min][2]], [V[min][3], V[min][4], V[min][5]], [V[min][6], V[min][7], V[min][8]]])
    U, D, V = np.linalg.svd(F, full_matrices = True)
    D[np.argmin(D)] = 0
    D2 = np.zeros((3, 3))
    D2[0][0] = D[0]
    D2[1][1] = D[1]
    D2[2][2] = D[2]
    F = np.dot(U, D2)
    F = np.dot(F, V)
    T = np.array([[row_m, 0, -row * row_m], [0, col_m, -col * col_m], [0, 0, 1]])
    F = np.dot(T.transpose(), F)
    F = np.dot(F, T)
    return F

def compute_F_mine(M) :
    end_sum = 0
    min_temp = 10
    while True :
        start = time.time()
        list = []
        num = random.randrange(0, M.shape[0])
        for x in range(9) :
            while num in list :
                num = random.randrange(0, M.shape[0])
            list.append(num)
        B = []
        for x in range(0, 9) :
            x1 = M[list[x]][0]
            y1 = M[list[x]][1]
            x2 = M[list[x]][2]
            y2 = M[list[x]][3]
            A = [x1 * x2, x1 * y2, x1, y1 * x2, y1 * y2, y1, x2, y2, 1]
            B.append(A)
        B = np.array(B).astype(np.float64)
        U, D, V = np.linalg.svd(B, full_matrices=True)
        min = np.argmin(D)
        F = np.array([[V[min][0], V[min][1], V[min][2]], [V[min][3], V[min][4], V[min][5]], [V[min][6], V[min][7], V[min][8]]])
        temp = compute_avg_reproj_error(M, F)
        if (temp < min_temp) :
            min_temp = temp
            F_result = F
        end = time.time() - start
        end_sum = end_sum + end
        if (end_sum > 2.95) :
            return F_result

img10 = cv2.imread('temple1.png')
img20 = cv2.imread('temple2.png')
M = np.loadtxt('temple_matches.txt')

row = img10.shape[1] / 2
row_m = 2 / img10.shape[1]
col = img10.shape[0] / 2
col_m = 2 / img10.shape[0]

F = compute_F_raw(M)
a1 = compute_avg_reproj_error(M, F)
F = compute_F_norm(M)
a2 = compute_avg_reproj_error(M, F)
F = compute_F_mine(M)
a3 = compute_avg_reproj_error(M, F)
F1 = F

print("Average Reprojection Errors (temple1.png and temple2.png)")
print("   Raw =", a1)
print("   Norm =", a2)
print("   Mine =", a3)

img30 = cv2.imread('house1.jpg')
img40 = cv2.imread('house2.jpg')
M = np.loadtxt('house_matches.txt')

row = img30.shape[1] / 2
row_m = 2 / img30.shape[1]
col = img30.shape[0] / 2
col_m = 2 / img30.shape[0]

F = compute_F_raw(M)
a1 = compute_avg_reproj_error(M, F)
F = compute_F_norm(M)
a2 = compute_avg_reproj_error(M, F)
F = compute_F_mine(M)
a3 = compute_avg_reproj_error(M, F)
F2 = F

print("")
print("Average Reprojection Errors (house1.jpg and house2.jpg)")
print("   Raw =", a1)
print("   Norm =", a2)
print("   Mine =", a3)

img50 = cv2.imread('library1.jpg')
img60 = cv2.imread('library2.jpg')
M = np.loadtxt('library_matches.txt')

row = img50.shape[1] / 2
row_m = 2 / img50.shape[1]
col = img50.shape[0] / 2
col_m = 2 / img50.shape[0]

F = compute_F_raw(M)
a1 = compute_avg_reproj_error(M, F)
F = compute_F_norm(M)
a2 = compute_avg_reproj_error(M, F)
F = compute_F_mine(M)
a3 = compute_avg_reproj_error(M, F)
F3 = F

print("")
print("Average Reprojection Errors (library1.jpg and library2.jpg)")
print("   Raw =", a1)
print("   Norm =", a2)
print("   Mine =", a3)

while True :
    if keyboard.is_pressed('q') :
        break
    else :
        img1 = cv2.imread('temple1.png')
        img2 = cv2.imread('temple2.png')
        img3 = cv2.imread('house1.jpg')
        img4 = cv2.imread('house2.jpg')
        img5 = cv2.imread('library1.jpg')
        img6 = cv2.imread('library2.jpg')

        M = np.loadtxt('temple_matches.txt')

        p1 = random.randrange(0, M.shape[0])
        while True :
            p2 = random.randrange(0, M.shape[0])
            if p2 != p1 :
                break
        while True :
            p3 = random.randrange(0, M.shape[0])
            if p3 != p1 :
                if p3 != p2 :
                    break

        img1 = cv2.circle(img1, (int(M[p1][0]), int(M[p1][1])), 5, (0, 0, 255), -1)
        A1 = np.dot(np.array([M[p1][0], M[p1][1], 1]), F1)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img1.shape[1]), int((A1[0] * img1.shape[1] + A1[2]) / (-A1[1])))
        img1 = cv2.line(img1, p11, p12, (0, 0, 255), 2)

        img1 = cv2.circle(img1, (int(M[p2][0]), int(M[p2][1])), 5, (0, 255, 0), -1)
        A1 = np.dot(np.array([M[p2][0], M[p2][1], 1]), F1)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img1.shape[1]), int((A1[0] * img1.shape[1] + A1[2]) / (-A1[1])))
        img1 = cv2.line(img1, p11, p12, (0, 255, 0), 2)

        img1 = cv2.circle(img1, (int(M[p3][0]), int(M[p3][1])), 5, (255, 0, 0), -1)
        A1 = np.dot(np.array([M[p3][0], M[p3][1], 1]), F1)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img1.shape[1]), int((A1[0] * img1.shape[1] + A1[2]) / (-A1[1])))
        img1 = cv2.line(img1, p11, p12, (255, 0, 0), 2)

        img2 = cv2.circle(img2, (int(M[p1][2]), int(M[p1][3])), 5, (0, 0, 255), -1)
        A1 = np.dot(np.array([M[p1][2], M[p1][3], 1]), F1)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img2.shape[1]), int((A1[0] * img2.shape[1] + A1[2]) / (-A1[1])))
        img2 = cv2.line(img2, p11, p12, (0, 0, 255), 2)

        img2 = cv2.circle(img2, (int(M[p2][2]), int(M[p2][3])), 5, (0, 255, 0), -1)
        A1 = np.dot(np.array([M[p2][2], M[p2][3], 1]), F1)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img2.shape[1]), int((A1[0] * img2.shape[1] + A1[2]) / (-A1[1])))
        img2 = cv2.line(img2, p11, p12, (0, 255, 0), 2)

        img2 = cv2.circle(img2, (int(M[p3][2]), int(M[p3][3])), 5, (255, 0, 0), -1)
        A1 = np.dot(np.array([M[p3][2], M[p3][3], 1]), F1)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img2.shape[1]), int((A1[0] * img2.shape[1] + A1[2]) / (-A1[1])))
        img2 = cv2.line(img2, p11, p12, (255, 0, 0), 2)

        img12 = cv2.hconcat([img1, img2])
        cv2.imshow('temple', img12.astype('uint8'))

        M = np.loadtxt('house_matches.txt')

        p1 = random.randrange(0, M.shape[0])
        while True :
            p2 = random.randrange(0, M.shape[0])
            if p2 != p1 :
                break
        while True :
            p3 = random.randrange(0, M.shape[0])
            if p3 != p1 :
                if p3 != p2 :
                    break

        img3 = cv2.circle(img3, (int(M[p1][0]), int(M[p1][1])), 5, (0, 0, 255), -1)
        A1 = np.dot(np.array([M[p1][0], M[p1][1], 1]), F2)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img3.shape[1]), int((A1[0] * img3.shape[1] + A1[2]) / (-A1[1])))
        img3 = cv2.line(img3, p11, p12, (0, 0, 255), 2)

        img3 = cv2.circle(img3, (int(M[p2][0]), int(M[p2][1])), 5, (0, 255, 0), -1)
        A1 = np.dot(np.array([M[p2][0], M[p2][1], 1]), F2)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img3.shape[1]), int((A1[0] * img3.shape[1] + A1[2]) / (-A1[1])))
        img3 = cv2.line(img3, p11, p12, (0, 255, 0), 2)

        img3 = cv2.circle(img3, (int(M[p3][0]), int(M[p3][1])), 5, (255, 0, 0), -1)
        A1 = np.dot(np.array([M[p3][0], M[p3][1], 1]), F2)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img3.shape[1]), int((A1[0] * img3.shape[1] + A1[2]) / (-A1[1])))
        img3 = cv2.line(img3, p11, p12, (255, 0, 0), 2)

        img4 = cv2.circle(img4, (int(M[p1][2]), int(M[p1][3])), 5, (0, 0, 255), -1)
        A1 = np.dot(np.array([M[p1][2], M[p1][3], 1]), F2)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img4.shape[1]), int((A1[0] * img4.shape[1] + A1[2]) / (-A1[1])))
        img4 = cv2.line(img4, p11, p12, (0, 0, 255), 2)

        img4 = cv2.circle(img4, (int(M[p2][2]), int(M[p2][3])), 5, (0, 255, 0), -1)
        A1 = np.dot(np.array([M[p2][2], M[p2][3], 1]), F2)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img4.shape[1]), int((A1[0] * img4.shape[1] + A1[2]) / (-A1[1])))
        img4 = cv2.line(img4, p11, p12, (0, 255, 0), 2)

        img4 = cv2.circle(img4, (int(M[p3][2]), int(M[p3][3])), 5, (255, 0, 0), -1)
        A1 = np.dot(np.array([M[p3][2], M[p3][3], 1]), F2)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img4.shape[1]), int((A1[0] * img4.shape[1] + A1[2]) / (-A1[1])))
        img4 = cv2.line(img4, p11, p12, (255, 0, 0), 2)

        img34 = cv2.hconcat([img3, img4])
        cv2.imshow('house', img34.astype('uint8'))

        M = np.loadtxt('library_matches.txt')

        p1 = random.randrange(0, M.shape[0])
        while True :
            p2 = random.randrange(0, M.shape[0])
            if p2 != p1 :
                break
        while True :
            p3 = random.randrange(0, M.shape[0])
            if p3 != p1 :
                if p3 != p2 :
                    break

        img5 = cv2.circle(img5, (int(M[p1][0]), int(M[p1][1])), 5, (0, 0, 255), -1)
        A1 = np.dot(np.array([M[p1][0], M[p1][1], 1]), F3)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img5.shape[1]), int((A1[0] * img5.shape[1] + A1[2]) / (-A1[1])))
        img5 = cv2.line(img5, p11, p12, (0, 0, 255), 2)

        img5 = cv2.circle(img5, (int(M[p2][0]), int(M[p2][1])), 5, (0, 255, 0), -1)
        A1 = np.dot(np.array([M[p2][0], M[p2][1], 1]), F3)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img5.shape[1]), int((A1[0] * img5.shape[1] + A1[2]) / (-A1[1])))
        img5 = cv2.line(img5, p11, p12, (0, 255, 0), 2)

        img5 = cv2.circle(img5, (int(M[p3][0]), int(M[p3][1])), 5, (255, 0, 0), -1)
        A1 = np.dot(np.array([M[p3][0], M[p3][1], 1]), F3)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img5.shape[1]), int((A1[0] * img5.shape[1] + A1[2]) / (-A1[1])))
        img5 = cv2.line(img5, p11, p12, (255, 0, 0), 2)

        img6 = cv2.circle(img6, (int(M[p1][2]), int(M[p1][3])), 5, (0, 0, 255), -1)
        A1 = np.dot(np.array([M[p1][2], M[p1][3], 1]), F3)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img6.shape[1]), int((A1[0] * img6.shape[1] + A1[2]) / (-A1[1])))
        img6 = cv2.line(img6, p11, p12, (0, 0, 255), 2)

        img6 = cv2.circle(img6, (int(M[p2][2]), int(M[p2][3])), 5, (0, 255, 0), -1)
        A1 = np.dot(np.array([M[p2][2], M[p2][3], 1]), F3)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img6.shape[1]), int((A1[0] * img6.shape[1] + A1[2]) / (-A1[1])))
        img6 = cv2.line(img6, p11, p12, (0, 255, 0), 2)

        img6 = cv2.circle(img6, (int(M[p3][2]), int(M[p3][3])), 5, (255, 0, 0), -1)
        A1 = np.dot(np.array([M[p3][2], M[p3][3], 1]), F3)
        p11 = (0, int(-A1[2] / A1[1]))
        p12 = (int(img6.shape[1]), int((A1[0] * img6.shape[1] + A1[2]) / (-A1[1])))
        img6 = cv2.line(img6, p11, p12, (255, 0, 0), 2)

        img56 = cv2.hconcat([img5, img6])
        cv2.imshow('library', img56.astype('uint8'))

        cv2.waitKey(0)



        n = 0
        nn = 0
        b = 0
        row = int(len(data['val1']) / 128)
        print(row)
        lele = len(data['val1'])
        print(lele)
        D = np.zeros((row, 128))
        while True:
            D[b][n] = data['val1'][nn]
            n = n + 1
            nn = nn + 1
            if n == 128:
                if nn == lele:
                    break
                n = 0
                b = b + 1
                continue
        # D = np.float32(D)#128개 벡터 * feature개수
        """ret, label, center = cv2.kmeans(D, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        C[3 * idx] = center[0]
        C[3 * idx + 1] = center[1]
        C[3 * idx + 2] = center[2]"""
        print(count)
        count = count + 1
        spath = './KME' + bfile[count - 1] + '.txt'
        np.savetxt(spath, D)
        # np.concatenate((CC, D))
        if count == 3:
            break
        data = np.fromfile('./sift/' + bfile[count], type)

    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    # ret,label,center=cv2.kmeans(Z,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    # CC = np.empty((1,128))
    # sift = cv2.xfeatures2d.SIFT_create(