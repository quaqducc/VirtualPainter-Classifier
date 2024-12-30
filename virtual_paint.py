import cv2
import numpy as np
import os
import torch

import HandTrackingModule as htm
from test_model import create_tensor_mask, classifier



# create camera frame
cap = cv2.VideoCapture(0)
width = 640
height = 640
cap.set(3,width)
cap.set(4,height)

# Import menu photo
folderPath = 'menu'
menulist = os.listdir(folderPath)
print(menulist)

# menu tab
menu = []
for img_path in menulist:
    image = cv2.imread(f'{folderPath}/{img_path}')
    menu.append(image)
    
# create header
header = menu[1]
header_height = 100
header = cv2.resize(header, (width, header_height))

# config painter
selection_mode_color = (255,0,255) 
draw_pointer_color = (0,255,0)
draw_color = (0, 255, 0)

xp,yp = 0,0

brush_thickness = 15
eraser_thickness = 50

draw = False
eraser = False

# initial hand detector
detector = htm.handDetector(detectionCon=0.8)

# create canvas for drawing
canvas_width = 480
img_canvas = np.zeros((canvas_width, height,3),np.uint8)

# detect class
CLASSES = ["apple", "book", "bowtie", "candle", "cloud", "cup", "door", "envelope", "eyeglasses", "guitar", "hammer",
           "hat", "ice cream", "leaf", "scissors", "star", "t-shirt", "pants", "lightning", "tree"]

while True:
    # 1. Import image
    success, img = cap.read()
    img = cv2.flip(img,1)
    
    # 2. Find Hand Landmarks
    img = detector.findHands(img)
    lmList = detector.findPosition(img,draw = False)
    
    if len(lmList) != 0: 
        # print(lmList)
        
        # tip of index and middle finger
        x1,y1 = lmList[8][1:]
        x2,y2 = lmList[12][1:]
        
    
    
    # 3. Check which fingers are up
    
    fingers = detector.fingerUp()
    #print(fingers)
    
    # 4. If Selection mode - Two finger are up

    if fingers[1] and fingers[2]:
        # print("Selection mode")
        xp,yp = 0,0
        if y1 < 100:
            if 280 < x1 < 360 and len(menu) > 1:
                header = menu[1]
                draw_color = (0, 255, 0)

                # draw =  True
                # eraser = False
            if 420 < x1 and len(menu) >2:
                draw_color = (0,0,0)
                header = menu[2] 

                # eraser = True
                # draw = False
                
        cv2.rectangle(img, (x1,y1-25),(x2,y2+25),selection_mode_color,cv2.FILLED)
        
    
    
    
    # 5. If Drawing mode - One finger is up
    if fingers[1] and not fingers[2]:
        # print("Drawing mode")

        cv2.circle(img, (x1, y1), 15, draw_color, cv2.FILLED)
        
        if xp == 0 and yp == 0:
            xp,yp =x1,y1
            
        if draw_color == (0,0,0):
            cv2.line(img, (xp,yp),(x1, y1), draw_color, eraser_thickness)
            cv2.line(img_canvas, (xp,yp),(x1, y1), draw_color, brush_thickness)
        else:
            cv2.line(img, (xp,yp),(x1, y1), draw_color, brush_thickness)
            cv2.line(img_canvas, (xp,yp),(x1, y1), draw_color, brush_thickness)
        xp, yp = x1,y1
    
    img_gray =  cv2.cvtColor(img_canvas,cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(img_gray,127,255, cv2.THRESH_BINARY)
    mask_bgr = cv2.cvtColor(mask,cv2.COLOR_GRAY2BGR)
    
    canvas_draw_only = cv2.bitwise_and(img_canvas, mask_bgr)
    img_combined = cv2.addWeighted(img, 1.0, canvas_draw_only, 1.0, 0)
    
        
    # Setting header
    header = cv2.resize(header, (640, 100))
    img_combined[0:100, 0:640] = header
    #img = cv2.addWeighted(img, 1, mask, 1, 0)
    cv2.imshow('Virtual Paint',img_combined)
    # cv2.imshow('Mask',mask)
    # cv2.imshow('Canvas',img_canvas)
    # cv2.imshow('Inv',img_inv)
    key = cv2.waitKey(1)

    # 6. Classification picture - Finish Drawing Mode
    if key == ord('s'):
        print("Classification mode")

        # Resize the mask to match the input size of the QuickDraw model
        prediction = classifier(mask, model_path='D:/HUST/Project1/Camera Draw/train/trained_models/13epochs.pth')
        predicted_class = CLASSES[prediction]
        print(predicted_class)

        result_frame = 255 * np.ones((200, 400, 3), dtype=np.uint8)  # Khung trắng để hiển thị
        cv2.putText(result_frame, f'Predicted Class:', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.putText(result_frame, f'{predicted_class}', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow("Prediction Result", result_frame)
        # cv2.putText(img, f'Predicted Class: {predicted_class}', (600, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    elif key == ord('q'):  # Nếu nhấn phím 'q', thoát
        break