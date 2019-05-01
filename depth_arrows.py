import imutils
import cv2
import numpy as np
from signdetector import SignDetector
import matplotlib.pyplot as plt

import sys
# sys.path.append('/home/jenglish/school/grad/ee6663/arrow-sign-classification/ModelClassifier')
# from ArrowClassifier import ArrowClassifier

sys.path.append('/home/jenglish/classes/ee6663/learning-classifier/keras-tutorial1')
import NNPredict

def main():
    wait = 20 #speed of video

    directions = []
    confidences = []
    times = []
    
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

                    # draw the contour and center of the shape on the image
                    if shape == "square":

                        appX, appY = [], []
                        
                        for pt in approx:
                            appX.append(pt[0][0])
                            appY.append(pt[0][1])

                        sign_ri = frame1[min(appY):max(appY), min(appX):max(appX)]

                        direction, confidence, classification_time = NNPredict.predict(sign_ri)

                        directions.append(direction)
                        confidences.append(confidence)
                        times.append(classification_time)

                        print(direction, confidence, classification_time)

                        # sri_width = int(sign_ri.shape[1] * 3)
                        # sri_height = int(sign_ri.shape[0] * 3)
                        # sri_dim = (sri_width, sri_height)
                        # # resize image
                        # sri_resize = cv2.resize(sign_ri, dim, interpolation = cv2.INTER_AREA) 


                        # cv2.imshow('Sign RI', sri_resize)
                        # cv2.waitKey(0)

                        # # convert the resized image to grayscale, blur it slightly,
                        # # and threshold it
                        # sri_gray = cv2.cvtColor(sri_resize, cv2.COLOR_BGR2GRAY)
                        # cv2.imshow("gray", sri_gray)
                        # sri_blurred = cv2.GaussianBlur(sri_gray, (9, 9), 0)
                        # cv2.imshow("blur", sri_blurred)
                        # sri_thresh = cv2.threshold(sri_blurred, 60, 255, cv2.THRESH_BINARY)[1]
                        # cv2.imshow("threshold", sri_thresh)

                        # kernel = np.ones((6,6), np.uint8) 
                        # sri_dilate = cv2.dilate(sri_thresh, kernel, iterations=3)
                        # cv2.imshow("dilate", sri_dilate)

                        # # find contours in the thresholded image and initialize the
                        # # shape detector
                        # sri_cnts = cv2.findContours(sri_dilate.copy(), cv2.RETR_EXTERNAL,
                        #     cv2.CHAIN_APPROX_SIMPLE)
                        # sri_cnts = imutils.grab_contours(sri_cnts)
                        
                        # ac = ArrowClassifier()
                        # direction = ''

                        # for c in sri_cnts:
                        #     # compute the center of the contour, then detect the name of the
                        #     # shape using only the contour
                        #     M = cv2.moments(c)
                        #     try:
                        #         cX = int((M["m10"] / M["m00"]) * ratio)
                        #         cY = int((M["m01"] / M["m00"]) * ratio)
                        #     except:
                        #         continue

                        #     try:
                        #         direction = ac.get_direction(c)
                        #         print(direction)
                        #     except ValueError:
                        #         continue

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

    correct = [] # confidence value
    correct_roi = [] # index number for roi
    incorrect = [] # confidence value
    incorrect_roi = [] # index number for roi

    for i in range(len(directions)):
        if directions[i] == 'right':
            correct.append(confidences[i])
            correct_roi.append(i)
        else:
            incorrect.append(confidences[i])
            incorrect_roi.append(i)

    # confidence - graph results
    # correct direction - blue
    plt.scatter(correct_roi, correct, color="blue", label='Correct Classification')

    # incorrect direction - red
    # for i in range(len(incorrect)):
    plt.scatter(incorrect_roi, incorrect, color="red", label='Incorrect Classification')

    plt.axis('equal')
    plt.ylim(0, 100)
    plt.xlabel('Region of Interest Frame Number', fontsize=18)
    plt.ylabel('Classification Confidence (%)', fontsize=16)
    plt.legend()
    plt.title('NN Classification Confidence per ROI Frame')
    plt.show()

    # mean correct classification confidence
    mccc = sum(correct) / len(correct)
    print('mean correct classification confidence (%): ', mccc)

    # time - graph results
    plt.plot([i for i in range(len(times))], [(t * 1000) for t in times], color="green")

    # mean classification time
    mct = (sum(times) * 1000) / len(times)
    print('mean correct classification confidence (ms): ', mct)

    # plt.axis('equal')
    # plt.ylim(0, 100)
    plt.xlabel('Region of Interest Frame Number', fontsize=18)
    plt.ylabel('Classification Time (ms)', fontsize=16)
    plt.title('NN Classification Time per ROI Frame')
    plt.show()
    
main()    
