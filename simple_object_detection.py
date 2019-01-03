import cv2
import numpy as np


def nothing(x):
    pass

# Initialize to webcame to be able for recording video from your laptop
cap = cv2.VideoCapture(0)

'''
we are going to create a trackbar to control the intensity or the value
of the HSV color within our video realtime. so we can determine which value
would be the best one to be able to detect and track the object
'''
# give the trackbar window a name
cv2.namedWindow('Trackbars')

# create the tracbar
# for lower value
cv2.createTrackbar('L-H', 'Trackbars', 0, 180, nothing)
cv2.createTrackbar('L-S', 'Trackbars', 126, 255, nothing)
cv2.createTrackbar('L-V', 'Trackbars', 145, 180, nothing)

# for the upper value
cv2.createTrackbar('U-H', 'Trackbars', 180, 180, nothing)
cv2.createTrackbar('U-S', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('U-V', 'Trackbars', 255, 255, nothing)

# do a loop for recording
while True:
    # read the webcam video as frame
    _, frame = cap.read()

    # convert our BGR color to HSV
    '''
    Note that opencv will only read the BGR value, not RGB.
    so we have to convert from BGR to HSV
    '''
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # get the trackbar value
    # for the lower value
    l_h = cv2.getTrackbarPos('L-H', 'Trackbars')
    l_s = cv2.getTrackbarPos('L-S', 'Trackbars')
    l_v = cv2.getTrackbarPos('L-V', 'Trackbars')

    # for the upper value
    u_h = cv2.getTrackbarPos('U-H', 'Trackbars')
    u_s = cv2.getTrackbarPos('U-S', 'Trackbars')
    u_v = cv2.getTrackbarPos('U-V', 'Trackbars')

    '''
    we are going to create our own mask
    first, we initialize the lower and the upper to create a range
    using cv2.inRange method
    '''

    lower_red = np.array([l_h, l_s, l_v])
    upper_red = np.array([u_h, u_s, u_v])
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # we filter out the noise using erode
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel)

    # then we find the countours in the mask
    (_, con, _) = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # we create rectangle box to track the object
    for cnt in con:
        (x, y, w, h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame, (int(x + w), int(y+h)), (int(x), int(y)), (0,255,0), 2)
    
    # we then see the result of the frame and the mask look like
    cv2.imshow('frame', frame)
    cv2.imshow('mask', mask)
    
    # stop recording by typing 'q'
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

# done
cap.release()
cv2.destroyAllWindows()