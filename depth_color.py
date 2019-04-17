import imutils
import cv2
import numpy as np

def main():
    wait = 20 #speed of video
    
    #first video: identify orange cones using colorVideo
    #shows original beside cone detection
    cap1 = cv2.VideoCapture('../colorVideo.mp4') #hard-coded input video

    if (cap1.isOpened() == False): 
        print("Unable to read camera feed")

    while(cap1.isOpened()):
        ret, frame = cap1.read()

        if ret == True:
            frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hsv_cones = cv2.cvtColor(frame1, cv2.COLOR_RGB2HSV)
            
            v = hsv_cones[:, :, 2]
            v = np.where(v <= 255 - 15, v + 15, 255)
            hsv_cones[:, :, 2] = v
            
            light_orange = (1, 190, 200)
            dark_orange = (18, 255, 255)
            mask1 = cv2.inRange(hsv_cones, light_orange, dark_orange)
            result1 = cv2.bitwise_and(hsv_cones, hsv_cones, mask=mask1)
            cv2.imshow('frame', np.hstack([frame, result1]))
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        else:
            break

    cap1.release()
    cv2.destroyAllWindows()
    
    #second videos: show colorVideo beside depthVideo
    #shows the difference in perspective between the two videos
    cap1 = cv2.VideoCapture('../colorVideo.mp4') #hard-coded input video
    cap2 = cv2.VideoCapture('../depthVideo.mp4') #hard-coded input video
    
    if (cap1.isOpened() == False or cap2.isOpened() == False): 
        print("Unable to read camera feed")
        
    while(cap1.isOpened() and cap2.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == True and ret2 == True:            
            cv2.imshow('frame1', frame1)
            cv2.imshow('frame2', frame2)
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        else:
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    
    #third videos: show capVideo beside depthVideo
    #shows the difference in perspective between the two videos
    cap1 = cv2.VideoCapture('../capVideo.mp4') #hard-coded input video
    cap2 = cv2.VideoCapture('../depthVideo.mp4') #hard-coded input video
    
    if (cap1.isOpened() == False or cap2.isOpened() == False): 
        print("Unable to read camera feed")
        
    while(cap1.isOpened() and cap2.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == True and ret2 == True:            
            cv2.imshow('frame1', frame1)
            cv2.imshow('frame2', frame2)
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        else:
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    
    #fourth videos: detect orange cones using capVideo
    #reverse RGB to HSV and detect blue
    cap1 = cv2.VideoCapture('../capVideo.mp4') #hard-coded input video
        
    if (cap1.isOpened() == False): 
        print("Unable to read camera feed")
        
    while(cap1.isOpened()):
        ret1, frame1 = cap1.read()

        if ret1 == True:
            hsv_cones = cv2.cvtColor(frame1, cv2.COLOR_RGB2HSV)
            
            v = hsv_cones[:, :, 2]
            v = np.where(v <= 255 - 10, v + 10, 255)
            hsv_cones[:, :, 2] = v           
            
            light = (110, 50, 50)
            dark = (160, 255, 255)
            mask1 = cv2.inRange(hsv_cones, light, dark)
            result1 = cv2.bitwise_and(hsv_cones, hsv_cones, mask=mask1)
            
            cv2.imshow('frame1', np.hstack([result1, hsv_cones]))
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        else:
            break

    cap1.release()
    cv2.destroyAllWindows()

    #fifth videos: show capVideo cone detection beside depthVideo edges
    cap1 = cv2.VideoCapture('../capVideo.mp4') #hard-coded input video
    cap2 = cv2.VideoCapture('../depthVideo.mp4') #hard-coded input video
    
    if (cap1.isOpened() == False or cap2.isOpened() == False): 
        print("Unable to read camera feed")
        
    while(cap1.isOpened() and cap2.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == True and ret2 == True:
            hsv_cones = cv2.cvtColor(frame1, cv2.COLOR_RGB2HSV)
            
            v = hsv_cones[:, :, 2]
            v = np.where(v <= 255 - 10, v + 10, 255)
            hsv_cones[:, :, 2] = v           
            
            light = (110, 50, 50)
            dark = (160, 255, 255)
            mask1 = cv2.inRange(hsv_cones, light, dark)
            result1 = cv2.bitwise_and(hsv_cones, hsv_cones, mask=mask1)
            
            blur = cv2.GaussianBlur(frame2,(13,13),0)
            edges = cv2.Canny(blur,100,200)
            
            cv2.imshow('frame1', result1)
            cv2.imshow('frame2', edges)
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        else:
            break

    cap1.release()
    cap2.release()    
    
main()
