import cv2 
import numpy as np
import time
if __name__ == '__main__':
    inputImage = cv2.imread("test.jpg",0)
    _, thr = cv2.threshold(inputImage, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("origin.jpg", thr)
    cv2.imshow("thr",thr)
    cv2.waitKey()

    kernel9 = np.ones((900,900), np.uint8)
    kernel3 = np.ones((3,3), np.uint8)

    start = time.time()
    res1 = cv2.erode(thr, kernel3)#侵蝕
    print("--------------------- 侵蝕: kernel(3,3): {0} -----------------------".format(time.time()- start))
    cv2.imwrite("kernel3.jpg", res1)
    cv2.waitKey()
    

    start = time.time() 
    res2 = cv2.erode(thr, kernel9)#侵蝕
    print("--------------------- 侵蝕: kernel(900,900){0} ---------------------".format(time.time()- start))
    cv2.imwrite("kernel9.jpg", res2)
    cv2.waitKey()
    
    
    
    start = time.time() 
    res3 = cv2.erode(thr, kernel3)#侵蝕
    print("--------------------- 侵蝕: kernel(3,3), 使用加速: {0} --------------".format(time.time()- start))
    cv2.imwrite("enable_optimize.jpg", res3)
    cv2.waitKey()

    cv2.setUseOptimized(False)#  停用加速

    start = time.time() 
    res4 = cv2.erode(thr, kernel3)#侵蝕
    print("--------------------- 侵蝕: kernel(3,3), 停用加速: {0} --------------".format(time.time()- start))
    cv2.imwrite("disable_optimize.jpg", res4)
    cv2.waitKey()