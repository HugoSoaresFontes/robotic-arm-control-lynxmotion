import cv2
import sys
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()

    print (ret)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
