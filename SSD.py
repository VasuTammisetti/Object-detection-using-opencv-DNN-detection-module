#importing required libraries
import numpy as np
import cv2
#cap = cv2.VideoCapture(0)
#loading the video to opencv library
#cap = cv2.VideoCapture('C:\\Users\\vasu0\Desktop\\SSD Object detection\\4K camera example for Traffic Monitoring (Road).mp4')
cap = cv2.VideoCapture('C:\\Users\\vasu0\Desktop\\SSD Object detection\\KITTI dataset sequence 07 video (360p).mp4')
#Setting image width and size with Ids(for width id=3 and heiht id=4)
cap.set(3,620)
cap.set(4,480)
#creating empty matrix to take class nems one by one from coco class nameset
classNames = []
#load the coco nameset
classFile = 'C:\\Users\\vasu0\\Desktop\\SSD Object detection\\coco.names'
#To read names one by one from classile and to write it to classnaes object
with open(classFile,'rt') as f:
    classNames = f.read().strip('\n').split('\n')
    print(classNames)
#To import configuration file and weights file to feed dnn(from Opencv documentations)
configPath= 'C:\\Users\\vasu0\\Desktop\\SSD Object detection\\ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
#pre trained weights 
weightsPath = 'C:\\Users\\vasu0\\Desktop\\SSD Object detection\\frozen_inference_graph.pb'
#Initiating DNN to detecct class type by passing weigts and configurations
net = cv2.dnn_DetectionModel(weightsPath,configPath)
# MobileNet requires fixed dimensions for input frames
# so we have to ensure that it is resized to 320*320 pixels
net.setInputSize(200,200)
# rescaling of different sized Images/Frames
net.setInputScale(1.0/127.5)
#normalizing the input by using mean subtraction
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)
#while True:
#Reading input video frame by frame using while loop
while(cap.isOpened()):
#while True:
    success, img = cap.read()
#setting threshhold value to identify required image by comparing feature map and bounding box overlap
#based on class id from dnn prediction we have to display image with that id along with bounding box around it
    classIds, confs, bbox = net.detect(img,confThreshold=0.4)
    print(classIds,bbox)
    if len(classIds) !=0:
#The network predocts id's from 1 but in our namelist id's start from 0 so we have to subtract 1 from class id
#class  ,confidence ,bbox fitting to predicted image
        for classId, confidence, box in zip(classIds.flatten(),confs.flatten(),bbox):
            cv2.rectangle(img,box,color=(200,5,250),thickness=2) # Green box
#Putting text on bounding box
            cv2.putText(img,f'{classNames[classId-1]} {int(confidence * 100)}%',(box[0]+10,box[1]+30),cv2.FONT_HERSHEY_PLAIN,2,(0,200,20),2)



#Displaying output
    cv2.imshow('image',img)
    cv2.waitKey(1)
