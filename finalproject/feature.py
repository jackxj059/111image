import numpy as np
import matplotlib.pyplot as plt
import cv2 
import os 
# 建立 3D 圖形
fig = plt.figure()
ax = fig.gca(projection='3d')
path = "sample"
# 產生 3D 座標資料
marker=["x","o"]
color = ["b","c"]
for index, category in enumerate(os.listdir(path)):
    for picture in os.listdir(path+"/"+category):
        print("path", path+"/"+category)
        pic = cv2.imread(path+"/"+category+"/"+ picture)

        hsv = cv2.cvtColor(pic, cv2.COLOR_BGR2HSV)
        shape = pic.shape
        length = 0
        for px in range(0, shape[0]-1,1):
            for py in range(0,shape[1]-1,1):
                length = length + 1
                z = hsv[px][py][0]
                x = hsv[px][py][1]
                y = hsv[px][py][2]
                c = 100
                if length > 5:
                    break
        # 繪製 3D 座標點

                ax.scatter(x, y, z, c=color[index], cmap='Reds', marker=marker[index] )

# 顯示圖例
ax.legend()

# 顯示圖形
plt.show()