import imutils
import cv2

def main():
    wait = 20 #speed of video
    
    #1st video: original, no effects
    
    cap = cv2.VideoCapture('../depthVideo.mp4') #hard-coded input video

    if (cap.isOpened() == False): 
        print("Unable to read camera feed")

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            cv2.imshow('frame',frame)
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    
    #2nd video: original converted to grayscale
    
    cap = cv2.VideoCapture('../depthVideo.mp4')

    if (cap.isOpened() == False): 
        print("Unable to read camera feed")

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            cv2.imshow('frame',gray)
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    
    #3rd video: edge detection applied to original, no other effects
    
    cap = cv2.VideoCapture('../depthVideo.mp4')

    if (cap.isOpened() == False): 
        print("Unable to read camera feed")

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            edges = cv2.Canny(frame,150,300)
            cv2.imshow('frame',edges)
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    
    cap = cv2.VideoCapture('../depthVideo.mp4')
    
    #4th video: Gaussian blur

    if (cap.isOpened() == False): 
        print("Unable to read camera feed")

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            blur = cv2.GaussianBlur(frame,(13,13),0)
            cv2.imshow('frame',blur)
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    
    #5th video: edge detection with Gaussian blur
    
    cap = cv2.VideoCapture('../depthVideo.mp4')

    if (cap.isOpened() == False): 
        print("Unable to read camera feed")

    while(cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            blur = cv2.GaussianBlur(frame,(13,13),0)
            edges = cv2.Canny(blur,100,200)
            cv2.imshow('frame',edges)
            if cv2.waitKey(wait) & 0xFF == ord('q'):
                break
        else:
            break

    cap.release()
    
    
    
main()
