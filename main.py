#All the imports go here
#from typing import Dict

import cv2
import numpy as np
import mediapipe as mp
from collections import deque

def setValues(x):
    print("")

# Creating trackbars needed for adjusting the marker color
cv2.namedWindow("Color Detector")
cv2.createTrackbar("Upper Hue","Color Detector",153,180,setValues)
cv2.createTrackbar("Upper Saturation","Color Detector",255,255,setValues)
cv2.createTrackbar("Upper Value","Color Detector",255,255,setValues)
cv2.createTrackbar("Lower Hue","Color Detector",64,100,setValues)
cv2.createTrackbar("Lower Saturation","Color Detector",171,255,setValues)
cv2.createTrackbar("Lower Value","Color Detector",78,255,setValues)

# Giving different arrays to handle color points of different color
bpoints = [deque(maxlen=1024)]
gpoints = [deque(maxlen=1024)]
rpoints = [deque(maxlen=1024)]
ypoints = [deque(maxlen=1024)]

# These index will be used to mark the points in particular arrays of specififc color
blue_index = 0
green_index = 0
red_index = 0
yellow_index = 0

# The kernel to be used for dilation purpose

kernel = np.ones((5,5),np.uint8)

colors = [(255,0,0),(0,255,0),(0,0,255),(0,255,255)]
colorIndex = 0

# Code for canvas setup
paintWindow = np.zeros((471,636,3))+255
paintWindow = cv2.rectangle(paintWindow,(40,1),(140,65),(0,0,0),2)
paintWindow = cv2.rectangle(paintWindow,(160,1),(255,65),colors[0],1)
paintWindow = cv2.rectangle(paintWindow,(275,1),(370,65),colors[1],1)
paintWindow = cv2.rectangle(paintWindow,(390,1),(485,65),colors[2],1)
paintWindow = cv2.rectangle(paintWindow,(505,1),(600,65),colors[3],1)

cv2.putText(paintWindow, "Clear", (49,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),2,cv2.LINE_AA)
cv2.putText(paintWindow, "Blue", (185,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
cv2.putText(paintWindow, "Green", (298,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
cv2.putText(paintWindow, "Red", (420,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),2,cv2.LINE_AA)
cv2.putText(paintWindow, "Yellow", (520,33),cv2.FONT_HERSHEY_SIMPLEX,0.5,(150,150,150),2,cv2.LINE_AA)
cv2.namedWindow('Paint',cv2.WINDOW_AUTOSIZE)

# Open Webcam

cap = cv2.VideoCapture(0)

while True:
    ret , frame = cap.read()

    frame = cv2.flip(frame,1)
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    u_hue = cv2.getTrackbarPos("Upper Hue", "Color Detector")
    u_saturation = cv2.getTrackbarPos("Upper Saturation", "Color Detector")
    u_value = cv2.getTrackbarPos("Upper Value", "Color Detector")
    l_hue = cv2.getTrackbarPos("Lower Hue", "Color Detector")
    l_saturation = cv2.getTrackbarPos("Lower Saturation", "Color Detector")
    l_value = cv2.getTrackbarPos("Lower Value", "Color Detector")
    Upper_hsv = np.array([u_hue,u_saturation,u_value])
    Lower_hsv = np.array([l_hue,l_saturation,l_value])

    #Adding color to the live frame

    frame = cv2.rectangle(frame,(40,1),(140,65),(122,122,122),1)
    frame = cv2.rectangle(frame, (160,1),(255,65),colors[0],1)
    frame = cv2.rectangle(frame,(275,1),(370,65),colors[1],1)
    frame = cv2.rectangle(frame,(390,1),(485,65),colors[2],1)
    frame = cv2.rectangle(frame,(505,1),(600,65),colors[3],1)

    cv2.putText(paintWindow, "Clear", (49, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "Blue", (185, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "Green", (298, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "Red", (420, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(paintWindow, "Yellow", (520, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 2, cv2.LINE_AA)

    #Identifting the pointer
    Mask = cv2.inRange(hsv,Lower_hsv,Upper_hsv)
    Mask = cv2.erode(Mask,kernel,iterations=1)
    Mask = cv2.morphologyEx(Mask,cv2.MORPH_OPEN,kernel)
    Mask = cv2.dilate(Mask,kernel,iterations=1)

    #find contours for the pointer after identifying
    cnts,_ = cv2.findContours(Mask.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    center = None

    # If the contours are formed
    if len(cnts) > 0:
        #sorting the contour to find the biggest
        cnt = sorted(cnts,key=cv2.contourArea,reverse = True)[0]
        #Get radius of the enclosing circle aroung the contour
        ((x,y),radius) = cv2.minEnclosingCircle(cnt)
        #Draw the circle around the contour
        cv2.circle(frame,(int(x),int(y)),int(radius),(0,255,255),2)
        #Calculating the center of the detected contour
        # M = cv2.moments(cnt)
        # center = (float(M['m10'],M['m00']),float(M['m01']/M['m00']))


        # Assuming cnt is your contour
        M = cv2.moments(cnt)

        # Calculate the centroid (center of mass)
        if M['m00'] != 0:
            center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
        else:
            # Handle division by zero if the contour has no area
            center = (0, 0)

        # Now, center variable holds the coordinates of the centroid

        #Now chechking if the user wants to click any button
        if center[1]<=65:
            if(40<=center[0]<=140): #Clear Button
                bpoints = [deque(maxlen=512)]
                gpoints = [deque(maxlen=512)]
                rpoints = [deque(maxlen=512)]
                ypoints = [deque(maxlen=512)]

                blue_index = 0
                green_index = 0
                red_index = 0
                yellow_index = 0

                paintWindow[67:,:,:]=255
            elif 160 <= center[0] <= 255: # blue
                colorIndex = 0
            elif 275 <= center[0] <= 370: # green
                colorIndex = 1
            elif 390 <= center[0] <= 485: # red
                colorIndex = 2
            elif 505 <= center[0] <= 600: # yellow
                colorIndex = 3

        else:
            if colorIndex == 0:
                bpoints[blue_index].appendleft(center)
            elif colorIndex == 1:
                gpoints[green_index].appendleft(center)
            elif colorIndex == 2:
                rpoints[red_index].appendleft(center)
            elif colorIndex == 3:
                ypoints[yellow_index].appendleft(center)
        # Append the next deques when nothing is detected to avoid messing up
    else:
        bpoints.append(deque(maxlen=512))
        blue_index += 1
        gpoints.append(deque(maxlen=512))
        green_index += 1
        rpoints.append(deque(maxlen=512))
        red_index += 1
        ypoints.append(deque(maxlen=512))
        yellow_index += 1

    # Draw lines of all the colors on the canvas and frame
    points = [bpoints, gpoints, rpoints, ypoints]
    for i in range(len(points)):
        for j in range(len(points[i])):
            for k in range(1, len(points[i][j])):
                if points[i][j][k - 1] is None or points[i][j][k] is None:
                    continue
                cv2.line(frame, points[i][j][k - 1], points[i][j][k], colors[i], 2)
                cv2.line(paintWindow, points[i][j][k - 1], points[i][j][k], colors[i], 2)

    #Show all the Windows
    cv2.imshow("Tracking",frame)
    cv2.imshow("Paint",paintWindow)
    cv2.imshow("Mask",Mask)

    #Press Q to exit
    if cv2.waitKey(4) & 0xFF == ord("q"):
        break

#relaese all
cap.release()
cv2.destroyAllWindows()