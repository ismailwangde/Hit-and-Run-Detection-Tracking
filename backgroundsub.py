import cv2
import numpy as np

video = cv2.VideoCapture("C:/Users/wangd/Ismail/NMIMS/SemVII/Capstone/Final/capstone.mp4")

kernel = None

# Initialize the background subtractor and disable shadow detection
object_bg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Apply background subtraction
    fgmask = object_bg.apply(frame)
   
    # Threshold to remove any residual shadows (pure white mask)
    _, fgmask = cv2.threshold(fgmask, 250, 255, cv2.THRESH_BINARY)
    
    # Apply erosion and dilation to clean up the mask
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    fgmask = cv2.dilate(fgmask, kernel, iterations=2)
    
    # Extract the foreground (background subtracted part)
    foregroundPart = cv2.bitwise_and(frame, frame, mask=fgmask)
        
    # Display the background-subtracted (foreground) frame
    cv2.imshow('Background Subtracted Frame', cv2.resize(foregroundPart, None, fx=0.6, fy=0.6))
   
    # Wait for key press (check if 'q' is pressed to exit)
    k = cv2.waitKey(1) & 0xff
    if k == ord('q'):
        break

# Release the VideoCapture object
video.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
