import cv2
import numpy as np
from skimage.feature import local_binary_pattern
class LBP():
    def __init__(self, radius =1, n_points =8, winSize=10 ):
        self.sample = None
        self.describe = None         
        self.block =None
        self.radius = radius 
        self.lbp = None
        self.n_points = n_points
        self.winSize= winSize
    def setLBPImageScikit(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp = local_binary_pattern(gray_image, self.n_points, self.radius)
        lbp = np.array(lbp, dtype=np.uint8)
        self.lbp = lbp
        return  lbp
    
    def setLBPimage(self, image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imgLBP = np.zeros_like(gray_image)
        neighboor = 3 
        for ih in range(0,image.shape[0] - neighboor):
            for iw in range(0,image.shape[1] - neighboor):
                ### Step 1: 3 by 3 pixel
                img          = gray_image[ih:ih+neighboor,iw:iw+neighboor]
                center       = img[1,1]
                img01        = (img >= center)*1.0
                img01_vector = img01.T.flatten()
                img01_vector = np.delete(img01_vector,4)
                where_img01_vector = np.where(img01_vector)[0]
                if len(where_img01_vector) >= 1:
                    num = np.sum(2**where_img01_vector)
                else:
                    num = 0
                imgLBP[ih+1,iw+1] = num
        self.lbp = imgLBP
        return imgLBP

    def calSimilarity(self, sampleLocation, selectLocation):
        sample = self.lbp[sampleLocation[1]:sampleLocation[1]+ self.winSize, sampleLocation[0]:sampleLocation[0]+ self.winSize]
        sample_hist = cv2.calcHist([sample], [0], None, [256], [0, 256])
        sample_std = np.std(sample_hist)


        select = self.lbp[selectLocation[1]:selectLocation[1]+ self.winSize, selectLocation[0]:selectLocation[0]+ self.winSize ]
        select_hist = cv2.calcHist([select], [0], None, [256], [0, 256])
        select_std = np.std(select_hist)

        if select_std > sample_std :
            result = sample_std/select_std
        else:
            result = select_std/sample_std
        return result
    def scan(self, sampleLocation, thr = 0.95):
        result =np.zeros_like(self.lbp)
        width, heigh = self.lbp.shape
        for y in range(0,heigh-self.winSize, self.winSize):
            for x in range(0,width-self.winSize, self.winSize):
                similar = self.calSimilarity(sampleLocation,(x,y))
                if similar >= thr:
                    result[y:y + self.winSize,x: x + self.winSize] = 255
        return result
    
if __name__ == '__main__':
    winSize = 30
    lbp = LBP(winSize = winSize)
    image = cv2.imread('./rockroad.png' )
    # lbp_scikit_image = lbp.setLBPImageScikit(image)
    lbp_image = lbp.setLBPImageScikit(image)
    cv2.imshow("lbp_image",lbp_image)
    cv2.waitKey()
    # lbp_image = lbp.setLBPimage(image)
    lbp_image = cv2.cvtColor(lbp_image, cv2.COLOR_GRAY2BGR)

    size = image.shape 
    select_mode = True
    sample_location = [0,0]
    select_location = [0,0]
    cv2.namedWindow('image')

    while True:
        image_draw = image.copy()
        lbp_draw = lbp_image.copy()
        similarity = lbp.calSimilarity(sample_location, select_location)
        cv2.putText(image_draw, str(similarity) , (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image_draw, "sample" , tuple(sample_location), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(image_draw, tuple(sample_location), (sample_location[0]+ winSize, sample_location[1]+ winSize), (0, 255, 0), 2)

        cv2.putText(lbp_draw, str(similarity) , (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(lbp_draw, "sample" , tuple(sample_location), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.rectangle(lbp_draw, tuple(sample_location), (sample_location[0]+ winSize, sample_location[1]+ winSize), (0, 255, 0), 2)



        cv2.putText(image_draw, "select" , tuple(select_location), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(image_draw, tuple(select_location), (select_location[0]+ winSize, select_location[1]+ winSize), (255, 0, 0), 2)
     
        cv2.putText(lbp_draw, "select" , tuple(select_location), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.rectangle(lbp_draw, tuple(select_location), (select_location[0]+ winSize, select_location[1]+ winSize), (255, 0, 0), 2)

        cv2.imshow("image", image_draw)
        cv2.imshow("lbp", lbp_draw)
        
        key = cv2.waitKey(0)
        if key == ord('q'):
            break
        if key == ord('e'):
            select_mode = not select_mode
        if key == ord('k'):
            markers = lbp.scan((sample_location[0],sample_location[1]), thr = 0.9)
            cv2.imshow("marker", markers)
            img = image.copy()
            ret, markers = cv2.connectedComponents(markers) 
            markers = cv2.watershed(image, markers)
            img[markers == -1] = (0, 0, 255)
            cv2.imshow("image", img)
            cv2.waitKey()
            res = cv2.watershed(image , markers)

        if key == ord('w'):
            if select_mode:
                if not sample_location[1] <= 0:
                    sample_location = [sample_location[0],sample_location[1] - 10]
            else:
                if not select_location[1] <= 0:
                    select_location = [select_location[0],select_location[1] - 10]
                
        if key == ord('a'):
            if select_mode:
                if not sample_location[0] <= 0:
                    sample_location = [sample_location[0] - 10, sample_location[1]]
            else:
                if not select_location[0] <= 0:
                    select_location = [select_location[0] - 10, select_location[1]]
        if key == ord('s'):
            if select_mode:
                if not sample_location[1] >= size[0] - winSize:
                    sample_location = [sample_location[0] ,sample_location[1]  + 10 ]
            else:
                if not select_location[1] >= size[0] - winSize:
                    select_location = [select_location[0] ,select_location[1] + 10 ]        
        if key == ord('d'):
            if select_mode:
                if not sample_location[0] >= size[1] - winSize:
                    sample_location = [sample_location[0] + 10 ,sample_location[1]]
            else:
                if not select_location[0] >= size[1] - winSize:
                    select_location = [select_location[0] + 10 ,select_location[1]]
