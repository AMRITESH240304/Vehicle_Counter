import cv2 as cv 
import numpy as np
import tensorflow as tf 
import keras
tf.keras.layers.BatchNormalization(
    axis=-1,
    momentum=0.99,
    epsilon=0.001,
    center=True,
    scale=True,
    beta_initializer='zeros',
    gamma_initializer='ones',
    moving_mean_initializer='zeros',
    moving_variance_initializer='ones',
    beta_regularizer=None,
    gamma_regularizer=None,
    beta_constraint=None,
    gamma_constraint=None,
)
# url = "https://www.youtube.com/watch?v=KBsqQez-O4w"
# video = pafy.new(url)
# best = video.getbest(preftype="mp4")

cap = cv.VideoCapture('video.mp4')
cv.namedWindow('youtube',cv.WINDOW_NORMAL)
cv.resizeWindow('youtube',1920,1080)
# cap.open(best.url)

min_heigth_rect = 80
min_width_rect = 80   #min width rectangle
count_line_positon = 550
algo = cv.bgsegm.createBackgroundSubtractorMOG()

def centre_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx = x+x1
    cy = y+y1

    return cx,cy

detect = []
offset = 6 #allowable error between pixel
counter_park = 0

while (True):
    ret,frame1 = cap.read()
    grey = cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey,(3,3),5)
    
    img_sub = algo.apply(blur)
    dilate = cv.dilate(img_sub,np.ones((5,5)))
    kernal = cv.getStructuringElement(cv.MORPH_ELLIPSE,(5,5))
    dilate_ada = cv.morphologyEx(dilate,cv.MORPH_CLOSE,kernal)
    dilate_ada = cv.morphologyEx(dilate_ada,cv.MORPH_CLOSE,kernal)
    counter,h = cv.findContours(dilate_ada,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
    
    cv.line(frame1,(25,count_line_positon),(1200,count_line_positon),(255,127,0),3)
    
    for (i,c) in enumerate(counter):
        (x,y,w,h) = cv.boundingRect(c)
        val_counter = (w>=min_width_rect) and (h>= min_heigth_rect)
        if not val_counter:
            continue
            
        cv.rectangle(frame1,(x,y),(x+w,y+h),(0,0,255),2)
        
        centre = centre_handle(x,y,w,h)
        detect.append(centre)
        cv.circle(frame1,centre,4,(0,0,255),-1)
        
        for (x,y) in detect:
            if y<(count_line_positon + offset) and y>(count_line_positon - offset):
                counter_park += 1
            cv.line(frame1,(25,count_line_positon),(1200,count_line_positon),(0,127,255),3)
            detect.remove((x,y))
            print("vehicle counter:"+str(counter_park))
        
    cv.putText(frame1,"VEHICLE COUNT: "+str(counter_park),(450,70),cv.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)
        
    #cv.imshow('detector',dilate_ada)        
    cv.imshow('youtube',frame1)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
    
cv.destroyAllWindows()
cap.release()
