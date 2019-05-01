#from: pyimagesearch tutorials
import cv2
 
class SignDetector:
    def __init__(self):
        pass 
    
    def detect(self, c):
        # initialize the shape name and approximate the contour
        shape = "unidentified"
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.04 * peri, True)

        if len(approx) == 4:
            # compute the bounding box of the contour and use the
            # bounding box to compute the aspect ratio
            (x, y, w, h) = cv2.boundingRect(approx)
            ar = w / float(h)
 
            # a square will have an aspect ratio that is approximately
            # equal to one, otherwise, the shape is a badsign
            #returns badsign for any other shape
            shape = "square" if ar >= 0.80 and ar <= 1.20 else "badsign"
        else:
            pass
            
        return shape, approx
