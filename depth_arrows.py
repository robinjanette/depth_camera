import imutils
import cv2
import numpy as np
from signdetector import SignDetector


import sys
sys.path.append('/home/jenglish/school/grad/ee6663/arrow-sign-classification/ModelClassifier')
from ArrowClassifier import ArrowClassifier

def main():
    wait = 20 #speed of video
    
    # #first video: original tuned cropping and resizing
    # cap1 = cv2.VideoCapture('./colorVideo_left.mp4') #hard-coded input video
    # cap2 = cv2.VideoCapture('./depthVideo_left.mp4') #hard-coded input video
    
    # if (cap1.isOpened() == False or cap2.isOpened() == False): 
    #     print("Unable to read camera feed")
        
    # while(cap1.isOpened() and cap2.isOpened()):
    #     ret1, frame1 = cap1.read()
    #     ret2, frame2 = cap2.read()

    #     if ret1 == True and ret2 == True:            
            
    #         #blur the depth video
    #         blur = cv2.GaussianBlur(frame2,(9,9),0)
            
    #         #crop the depth video
    #         cropped = blur[85:385, 108:508]
            
    #         #resize the depth video back to the same dimensions as the color video
    #         r = 640 / cropped.shape[1]
    #         dim = (640, int(cropped.shape[0] * r))
    #         resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
            
    #         #add the two images together            
    #         img = cv2.add(frame1, resized)
            
    #         #display
    #         cv2.imshow('frame1', img)
            
    #         if cv2.waitKey(wait) & 0xFF == ord('q'):
    #             break

    #     else:
    #         break

    # cap1.release()
    # cap2.release()
    # cv2.destroyAllWindows()
    
    #second video: left arrow video with corrected cropping and resizing
    # cap1 = cv2.VideoCapture('./colorVideo_left.mp4') #hard-coded input video
    # cap2 = cv2.VideoCapture('./depthVideo_left.mp4') #hard-coded input video
    
    # if (cap1.isOpened() == False or cap2.isOpened() == False): 
    #     print("Unable to read camera feed")
        
    # while(cap1.isOpened() and cap2.isOpened()):
    #     ret1, frame1 = cap1.read()
    #     ret2, frame2 = cap2.read()

    #     if ret1 == True and ret2 == True:            
            
    #         #blur the depth video
    #         blur = cv2.GaussianBlur(frame2,(9,9),0)
            
    #         #crop the depth video
    #         cropped = blur[78:378, 108:508]
            
    #         #resize the depth video back to the same dimensions as the color video
    #         r = 640 / cropped.shape[1]
    #         dim = (640, int(cropped.shape[0] * r))
    #         resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
            
    #         #add the two images together            
    #         img = cv2.add(frame1, resized)
            
    #         #display
    #         cv2.imshow('frame1', img)
            
    #         if cv2.waitKey(wait) & 0xFF == ord('q'):
    #             break

    #     else:
    #         break

    # cap1.release()
    # cap2.release()
    # cv2.destroyAllWindows()

    # #third video: forward arrow video with corrected cropping and resizing
    # cap1 = cv2.VideoCapture('./colorVideo_forward.mp4') #hard-coded input video
    # cap2 = cv2.VideoCapture('./depthVideo_forward.mp4') #hard-coded input video
    
    # if (cap1.isOpened() == False or cap2.isOpened() == False): 
    #     print("Unable to read camera feed")
        
    # while(cap1.isOpened() and cap2.isOpened()):
    #     ret1, frame1 = cap1.read()
    #     ret2, frame2 = cap2.read()

    #     if ret1 == True and ret2 == True:            
            
    #         #blur the depth video
    #         blur = cv2.GaussianBlur(frame2,(9,9),0)
            
    #         #crop the depth video
    #         cropped = blur[78:378, 108:508]
            
    #         #resize the depth video back to the same dimensions as the color video
    #         r = 640 / cropped.shape[1]
    #         dim = (640, int(cropped.shape[0] * r))
    #         resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
            
    #         #add the two images together            
    #         img = cv2.add(frame1, resized)
            
    #         #display
    #         cv2.imshow('frame1', img)
            
    #         if cv2.waitKey(wait) & 0xFF == ord('q'):
    #             break

    #     else:
    #         break

    # cap1.release()
    # cap2.release()
    # cv2.destroyAllWindows()
    
    # #fourth video: right arrow video with corrected cropping and resizing
    # cap1 = cv2.VideoCapture('./colorVideo_right.mp4') #hard-coded input video
    # cap2 = cv2.VideoCapture('./depthVideo_right.mp4') #hard-coded input video
    
    # if (cap1.isOpened() == False or cap2.isOpened() == False): 
    #     print("Unable to read camera feed")
        
    # while(cap1.isOpened() and cap2.isOpened()):
    #     ret1, frame1 = cap1.read()
    #     ret2, frame2 = cap2.read()

    #     if ret1 == True and ret2 == True:            
            
    #         #blur the depth video
    #         blur = cv2.GaussianBlur(frame2,(9,9),0)
            
    #         #crop the depth video
    #         cropped = blur[78:378, 108:508]
            
    #         #resize the depth video back to the same dimensions as the color video
    #         r = 640 / cropped.shape[1]
    #         dim = (640, int(cropped.shape[0] * r))
    #         resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
            
    #         #add the two images together            
    #         img = cv2.add(frame1, resized)
            
    #         #display
    #         cv2.imshow('frame1', img)
            
    #         if cv2.waitKey(wait) & 0xFF == ord('q'):
    #             break

    #     else:
    #         break

    # cap1.release()
    # cap2.release()
    # cv2.destroyAllWindows()
    
    # #fifth video: left arrow video with sign detection
    # cap1 = cv2.VideoCapture('./colorVideo_left.mp4') #hard-coded input video
    # cap2 = cv2.VideoCapture('./depthVideo_left.mp4') #hard-coded input video
    
    # if (cap1.isOpened() == False or cap2.isOpened() == False): 
    #     print("Unable to read camera feed")
        
    # while(cap1.isOpened() and cap2.isOpened()):
    #     ret1, frame1 = cap1.read()
    #     ret2, frame2 = cap2.read()

    #     if ret1 == True and ret2 == True:            
            
    #         #blur the depth video
    #         blur = cv2.GaussianBlur(frame2,(9,9),0)
            
    #         #crop the depth video
    #         cropped = blur[78:378, 108:508]
            
    #         #resize the depth video back to the same dimensions as the color video
    #         r = 640 / cropped.shape[1]
    #         dim = (640, int(cropped.shape[0] * r))
    #         resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
    #         gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #         thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
            
    #         cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         cnts = imutils.grab_contours(cnts)
    #         s = SignDetector()
            
    #         for c in cnts:
    #             # compute the center of the contour
    #             M = cv2.moments(c)
    #             if (M["m00"] > 0.0):
    #                 cX = int(M["m10"] / M["m00"])
    #                 cY = int(M["m01"] / M["m00"])
    #                 shape = SignDetector.detect(s, c)
 
    #                 # draw the contour and center of the shape on the image
    #                 if shape == "square":
    #                     cv2.drawContours(resized, [c], -1, (0, 255, 0), 2)
    #                     cv2.circle(resized, (cX, cY), 7, (255, 255, 255), -1)
    #                     cv2.putText(resized, "center", (cX - 20, cY - 20),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #             else:
    #                 break
            
    #         #add the two images together            
    #         img = cv2.add(frame1, resized)
            
    #         #display
    #         cv2.imshow('frame1', img)
            
    #         if cv2.waitKey(wait) & 0xFF == ord('q'):
    #             break

    #     else:
    #         break

    # cap1.release()
    # cap2.release()
    # cv2.destroyAllWindows()

    # #sixth video: forward arrow video with sign detection
    # cap1 = cv2.VideoCapture('./colorVideo_forward.mp4') #hard-coded input video
    # cap2 = cv2.VideoCapture('./depthVideo_forward.mp4') #hard-coded input video
    
    # if (cap1.isOpened() == False or cap2.isOpened() == False): 
    #     print("Unable to read camera feed")
        
    # while(cap1.isOpened() and cap2.isOpened()):
    #     ret1, frame1 = cap1.read()
    #     ret2, frame2 = cap2.read()

    #     if ret1 == True and ret2 == True:            
            
    #         #blur the depth video
    #         blur = cv2.GaussianBlur(frame2,(9,9),0)
            
    #         #crop the depth video
    #         cropped = blur[78:378, 108:508]
            
    #         #resize the depth video back to the same dimensions as the color video
    #         r = 640 / cropped.shape[1]
    #         dim = (640, int(cropped.shape[0] * r))
    #         resized = cv2.resize(cropped, dim, interpolation=cv2.INTER_AREA)
    #         gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #         thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
            
    #         cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #         cnts = imutils.grab_contours(cnts)
    #         s = SignDetector()
            
    #         for c in cnts:
    #             # compute the center of the contour
    #             M = cv2.moments(c)
    #             if (M["m00"] > 0.0):
    #                 cX = int(M["m10"] / M["m00"])
    #                 cY = int(M["m01"] / M["m00"])
    #                 shape = SignDetector.detect(s, c)
 
    #                 # draw the contour and center of the shape on the image
    #                 if shape == "square":
    #                     cv2.drawContours(resized, [c], -1, (0, 255, 0), 2)
    #                     cv2.circle(resized, (cX, cY), 7, (255, 255, 255), -1)
    #                     cv2.putText(resized, "center", (cX - 20, cY - 20),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    #             else:
    #                 break
            
    #         #add the two images together            
    #         img = cv2.add(frame1, resized)
            
    #         #display
    #         cv2.imshow('frame1', img)
            
    #         if cv2.waitKey(wait) & 0xFF == ord('q'):
    #             break

    #     else:
    #         break

    # cap1.release()
    # cap2.release()
    # cv2.destroyAllWindows()
    
    #seventh video: right arrow video with sign detection
    cap1 = cv2.VideoCapture('./colorVideo_right.mp4') #hard-coded input video
    cap2 = cv2.VideoCapture('./depthVideo_right.mp4') #hard-coded input video
    
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
                    shape, approx = SignDetector.detect(s, c)

                    print('shape',shape)
 
                    # draw the contour and center of the shape on the image
                    if shape == "square":

                        appX, appY = [], []
                        
                        for pt in approx:
                            appX.append(pt[0][0])
                            appY.append(pt[0][1])

                        sign_ri = frame1[min(appY):max(appY), min(appX):max(appX)]

                        sri_width = int(sign_ri.shape[1] * 3)
                        sri_height = int(sign_ri.shape[0] * 3)
                        sri_dim = (sri_width, sri_height)
                        # resize image
                        sri_resize = cv2.resize(sign_ri, dim, interpolation = cv2.INTER_AREA) 


                        cv2.imshow('Sign RI', sri_resize)
                        cv2.waitKey(0)

                        # convert the resized image to grayscale, blur it slightly,
                        # and threshold it
                        sri_gray = cv2.cvtColor(sri_resize, cv2.COLOR_BGR2GRAY)
                        cv2.imshow("gray", sri_gray)
                        sri_blurred = cv2.GaussianBlur(sri_gray, (9, 9), 0)
                        cv2.imshow("blur", sri_blurred)
                        sri_thresh = cv2.threshold(sri_blurred, 60, 255, cv2.THRESH_BINARY)[1]
                        cv2.imshow("threshold", sri_thresh)

                        kernel = np.ones((6,6), np.uint8) 
                        sri_dilate = cv2.dilate(sri_thresh, kernel, iterations=3)
                        cv2.imshow("dilate", sri_dilate)

                        # find contours in the thresholded image and initialize the
                        # shape detector
                        sri_cnts = cv2.findContours(sri_dilate.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
                        sri_cnts = imutils.grab_contours(sri_cnts)
                        
                        ac = ArrowClassifier()
                        direction = ''

                        for c in sri_cnts:
                            # compute the center of the contour, then detect the name of the
                            # shape using only the contour
                            M = cv2.moments(c)
                            try:
                                cX = int((M["m10"] / M["m00"]) * ratio)
                                cY = int((M["m01"] / M["m00"]) * ratio)
                            except:
                                continue

                            try:
                                direction = ac.get_direction(c)
                                print(direction)
                            except ValueError:
                                continue

                        cv2.putText(resized, direction, (cX - 20, cY - 20),
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
