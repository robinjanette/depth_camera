import imutils
import cv2
import numpy as np
from signdetector import SignDetector

def main():
    wait = 20 #speed of video
    
    #first video: original tuned cropping and resizing
    cap1 = cv2.VideoCapture('../colorVideo_left.mp4') #hard-coded input video
    cap2 = cv2.VideoCapture('../depthVideo_left.mp4') #hard-coded input video
    
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
    
    #second video: left arrow video with corrected cropping and resizing
    cap1 = cv2.VideoCapture('../colorVideo_left.mp4') #hard-coded input video
    cap2 = cv2.VideoCapture('../depthVideo_left.mp4') #hard-coded input video
    
    if (cap1.isOpened() == False or cap2.isOpened() == False): 
        print("Unable to read camera feed")
        
    while(cap1.isOpened() and cap2.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == True and ret2 == True:            
            
            #blur the depth video
            blur = cv2.GaussianBlur(frame2,(9,9),0)
            
            #crop the depth video
            cropped = blur[78:378, 108:508]
            
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

    #third video: forward arrow video with corrected cropping and resizing
    cap1 = cv2.VideoCapture('../colorVideo_forward.mp4') #hard-coded input video
    cap2 = cv2.VideoCapture('../depthVideo_forward.mp4') #hard-coded input video
    
    if (cap1.isOpened() == False or cap2.isOpened() == False): 
        print("Unable to read camera feed")
        
    while(cap1.isOpened() and cap2.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == True and ret2 == True:            
            
            #blur the depth video
            blur = cv2.GaussianBlur(frame2,(9,9),0)
            
            #crop the depth video
            cropped = blur[78:378, 108:508]
            
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
    
    #fourth video: right arrow video with corrected cropping and resizing
    cap1 = cv2.VideoCapture('../colorVideo_right.mp4') #hard-coded input video
    cap2 = cv2.VideoCapture('../depthVideo_right.mp4') #hard-coded input video
    
    if (cap1.isOpened() == False or cap2.isOpened() == False): 
        print("Unable to read camera feed")
        
    while(cap1.isOpened() and cap2.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == True and ret2 == True:            
            
            #blur the depth video
            blur = cv2.GaussianBlur(frame2,(9,9),0)
            
            #crop the depth video
            cropped = blur[78:378, 108:508]
            
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
    
    #fifth video: left arrow video with sign detection
    cap1 = cv2.VideoCapture('../colorVideo_left.mp4') #hard-coded input video
    cap2 = cv2.VideoCapture('../depthVideo_left.mp4') #hard-coded input video
    
    if (cap1.isOpened() == False or cap2.isOpened() == False): 
        print("Unable to read camera feed")
        
    while(cap1.isOpened() and cap2.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == True and ret2 == True:            
            
            #blur the depth video
            blur = cv2.GaussianBlur(frame2,(9,9),0)
            
            #crop the depth video
            cropped = blur[78:378, 108:508]
            
            #resize the depth video back to the same dimensions as the color video
            r = 640 / cropped.shape[1]
            dim = (640, int(cropped.shape[0] * r))
            resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
            
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            s = SignDetector()
            
            for c in cnts:
                # compute the center of the contour
                M = cv2.moments(c)
                if (M["m00"] > 0.0):
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    shape = SignDetector.detect(s, c)
 
                    # draw the contour and center of the shape on the image
                    if shape == "square":
                        cv2.drawContours(resized, [c], -1, (0, 255, 0), 2)
                        cv2.circle(resized, (cX, cY), 7, (255, 255, 255), -1)
                        cv2.putText(resized, "center", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    break
            
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

    #sixth video: forward arrow video with sign detection
    cap1 = cv2.VideoCapture('../colorVideo_forward.mp4') #hard-coded input video
    cap2 = cv2.VideoCapture('../depthVideo_forward.mp4') #hard-coded input video
    
    if (cap1.isOpened() == False or cap2.isOpened() == False): 
        print("Unable to read camera feed")
        
    while(cap1.isOpened() and cap2.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == True and ret2 == True:            
            
            #blur the depth video
            blur = cv2.GaussianBlur(frame2,(9,9),0)
            
            #crop the depth video
            cropped = blur[78:378, 108:508]
            
            #resize the depth video back to the same dimensions as the color video
            r = 640 / cropped.shape[1]
            dim = (640, int(cropped.shape[0] * r))
            resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
            
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            s = SignDetector()
            
            for c in cnts:
                # compute the center of the contour
                M = cv2.moments(c)
                if (M["m00"] > 0.0):
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    shape = SignDetector.detect(s, c)
 
                    # draw the contour and center of the shape on the image
                    if shape == "square":
                        cv2.drawContours(resized, [c], -1, (0, 255, 0), 2)
                        cv2.circle(resized, (cX, cY), 7, (255, 255, 255), -1)
                        cv2.putText(resized, "center", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    break
            
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
    
    #seventh video: right arrow video with sign detection
    cap1 = cv2.VideoCapture('../colorVideo_right.mp4') #hard-coded input video
    cap2 = cv2.VideoCapture('../depthVideo_right.mp4') #hard-coded input video
    
    if (cap1.isOpened() == False or cap2.isOpened() == False): 
        print("Unable to read camera feed")
        
    while(cap1.isOpened() and cap2.isOpened()):
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()

        if ret1 == True and ret2 == True:            
            
            #blur the depth video
            blur = cv2.GaussianBlur(frame2,(9,9),0)
            
            #crop the depth video
            cropped = blur[78:378, 108:508]
            
            #resize the depth video back to the same dimensions as the color video
            r = 640 / cropped.shape[1]
            dim = (640, int(cropped.shape[0] * r))
            resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
            
            cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = imutils.grab_contours(cnts)
            s = SignDetector()
            
            for c in cnts:
                # compute the center of the contour
                M = cv2.moments(c)
                if (M["m00"] > 0.0):
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    shape = SignDetector.detect(s, c)
 
                    # draw the contour and center of the shape on the image
                    if shape == "square":
                        cv2.drawContours(resized, [c], -1, (0, 255, 0), 2)
                        cv2.circle(resized, (cX, cY), 7, (255, 255, 255), -1)
                        cv2.putText(resized, "center", (cX - 20, cY - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                else:
                    break
            
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
    
main()    
