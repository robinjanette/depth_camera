import imutils
import cv2
import numpy as np

def main():
    wait = 20 #speed of video

    #first video: color video added to depth video
    cap1 = cv2.VideoCapture('../colorVideo.mp4') #hard-coded input video
    cap2 = cv2.VideoCapture('../depthVideo.mp4') #hard-coded input video
    
    if (cap1.isOpened() == False or cap2.isOpened() == False): 
        print("Unable to read camera feed")
        
    while(cap1.isOpened() and cap2.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == True and ret2 == True:            
            img = cv2.add(frame1, frame2)
            cv2.imshow('frame1', img)
            
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        else:
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

    #second video: color video added to blurred, cropped, and resized depth video
    cap1 = cv2.VideoCapture('../colorVideo.mp4') #hard-coded input video
    cap2 = cv2.VideoCapture('../depthVideo.mp4') #hard-coded input video
    
    if (cap1.isOpened() == False or cap2.isOpened() == False): 
        print("Unable to read camera feed")
        
    while(cap1.isOpened() and cap2.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == True and ret2 == True:            
            
            #blur the depth video
            blur = cv2.GaussianBlur(frame2,(9,9),0)
            
            #crop the depth video
            cropped = blur[85:385, 108:508]
            
            #resize the depth video back to the same dimensions as the color video
            r = 640 / cropped.shape[1]
            dim = (640, int(cropped.shape[0] * r))
            resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
            
            #add the two images together            
            img = cv2.add(frame1, resized)
            
            #display
            cv2.imshow('frame1', img)
            
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break

        else:
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    
    #third video: orange detection in color video added to blurred, cropped, and resized depth video
    cap1 = cv2.VideoCapture('../colorVideo.mp4') #hard-coded input video
    cap2 = cv2.VideoCapture('../depthVideo.mp4') #hard-coded input video
    
    if (cap1.isOpened() == False or cap2.isOpened() == False): 
        print("Unable to read camera feed")
        
    while(cap1.isOpened() and cap2.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == True and ret2 == True:            
            
            #blur the depth video
            blur = cv2.GaussianBlur(frame2,(9,9),0)
            
            #crop the depth video
            cropped = blur[85:385, 108:508]
            
            #resize the depth video back to the same dimensions as the color video
            r = 640 / cropped.shape[1]
            dim = (640, int(cropped.shape[0] * r))
            resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
            
            #detect orange cones
            hsv_cones = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)
            v = hsv_cones[:, :, 2]
            v = np.where(v <= 255 - 15, v + 15, 255)
            hsv_cones[:, :, 2] = v
            light_orange = (1, 190, 200)
            dark_orange = (18, 255, 255)
            mask1 = cv2.inRange(hsv_cones, light_orange, dark_orange)
            result1 = cv2.bitwise_and(hsv_cones, hsv_cones, mask=mask1)
            
            #add the two images together            
            img = cv2.add(result1, resized)
            
            #display
            cv2.imshow('frame1', img)
            
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break

        else:
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()
    
main()
