import numpy as np
import cv2

class Detector:
    def __init__(self, use_cuda=False):
        self.faceModel = cv2.dnn.readNetFromCaffe("models/res10_300x300_ssd_iter_140000.prototxt",
        caffeModel="models/res10_300x300_ssd_iter_140000.caffemodel")

        if use_cuda:
            self.faceModel.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self.faceModel.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    def processImage(self,imgName, path_to_img):
        try:
            self.img = cv2.imread(path_to_img+'/'+imgName)
            (self.height, self.width) = self.img.shape[:2]
            self.processFrame(imgName, path_to_img)
        except:
            print (imgName)

        #cv2.imshow("Output",self.img)
        #cv2.waitKey(0)

    def processFrame(self,imgName, path_to_img):
        blob = cv2.dnn.blobFromImage(self.img,1.0,(300,300),(104.0,177.0,123.0), swapRB = False, crop = False)

        self.faceModel.setInput(blob)
        predictions = self.faceModel.forward()
        arr = []
        for i in range(0, predictions.shape[2]):
            arr.append(predictions[0,0,i,2])
        if (max(arr)>0.5):
            cv2.imwrite(f'images_with_faces/'+str(max(arr))+'_'+imgName,self.img)