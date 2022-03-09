import numpy as np
import cv2
import datetime
import threading
import os
from Detector import *


        
def detector(img):
    detector = Detector(use_cuda=False)
    for i in range(1,100,1):
        detector.processImage(img)

def detectorCuda(names, path_to_img):
    detectorCuda = Detector(use_cuda=True)
    for name in names:
        detectorCuda.processImage(name, path_to_img)

if __name__ == '__main__':
    path_to_img = 'C:/Users/mihai/PycharmProjects/prnt_sc/img'
    names = os.listdir(path_to_img)
    npNames = np.array(names)
    npNamesSptited = np.array_split(npNames,10)
    #detectorCuda(npNamesSptited[0],path_to_img)
    
    startTimeCuda = datetime.datetime.now()
    thread0Cuda = threading.Thread(target=detectorCuda, args=(npNamesSptited[0],path_to_img,))
    thread1Cuda = threading.Thread(target=detectorCuda, args=(npNamesSptited[1],path_to_img,))
    thread2Cuda = threading.Thread(target=detectorCuda, args=(npNamesSptited[2],path_to_img,))
    thread3Cuda = threading.Thread(target=detectorCuda, args=(npNamesSptited[3],path_to_img,))
    thread4Cuda = threading.Thread(target=detectorCuda, args=(npNamesSptited[4],path_to_img,))
    thread5Cuda = threading.Thread(target=detectorCuda, args=(npNamesSptited[5],path_to_img,))
    thread6Cuda = threading.Thread(target=detectorCuda, args=(npNamesSptited[6],path_to_img,))
    thread7Cuda = threading.Thread(target=detectorCuda, args=(npNamesSptited[7],path_to_img,))
    thread8Cuda = threading.Thread(target=detectorCuda, args=(npNamesSptited[8],path_to_img,))
    thread9Cuda = threading.Thread(target=detectorCuda, args=(npNamesSptited[9],path_to_img,))
    thread0Cuda.start()
    thread1Cuda.start()
    thread2Cuda.start()
    thread3Cuda.start()
    thread4Cuda.start()
    thread5Cuda.start()
    thread6Cuda.start()
    thread7Cuda.start()
    thread8Cuda.start()
    thread9Cuda.start()
    thread0Cuda.join()
    thread1Cuda.join()
    thread2Cuda.join()
    thread3Cuda.join()
    thread4Cuda.join()
    thread5Cuda.join()
    thread6Cuda.join()
    thread7Cuda.join()
    thread8Cuda.join()
    thread9Cuda.join()
    endTimeCuda =  datetime.datetime.now()
    print('Cuda: '+str(endTimeCuda-startTimeCuda))
    
