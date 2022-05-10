import numpy as np
import cv2
import argparse

# video_file = "highway1.mp4"
# video_file = "highway2.mp4"
# video_file = "highway3.mp4"
# video_file = "highway4.mp4"
video_file = "highway5.mp4"


cap = cv2.VideoCapture(video_file)

# roi_mask = cap.read()[1]


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 20,
                       blockSize = 3 )

# Take first frame and find corners in it
ret, old_frame = cap.read()
roi_mask = old_frame
roi_mask[:, :540] = 0
roi_mask[:, 760:] = 0
roi_mask[:380, :] = 0
roi_mask[520:, :] = 0
# cv2.imshow("roimask",roi_mask)
# cv2.waitKey(0)
old_gray = cv2.cvtColor(roi_mask, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# check corners
for i in p0:
    x, y = i.ravel()
    old_frame = cv2.circle(roi_mask, (int(x), int(y)), 5, (0, 0, 255), -1)
cv2.imshow("old_frame",roi_mask)
#     old_frame = cv2.circle(old_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
# cv2.imshow("old_frame",old_frame)
cv2.waitKey(0)

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

feature_refresh_counter = 0

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break
    
    frame[:, :540] = 0
    frame[:, 760:] = 0
    frame[:380, :] = 0
    frame[520:, :] = 0
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    _, threshframe = cv2.threshold(gray_frame, 210, 255, cv2.THRESH_BINARY)

    graythresh = threshframe
    print(threshframe[600])


    cv2.imshow('frame', graythresh)

# calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, threshframe, p0, None, **lk_params)
    
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0,255,0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0,255,0), -1)
        frame = cv2.circle(frame, (int(c), int(d)), 5, (0,0,255), -1)
    img = cv2.add(frame, mask)
    # img = frame
    
    cv2.imshow('frame', img)

    key = cv2.waitKey(20)
    if key == ord("q"):
        break
    
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

    
cv2.destroyAllWindows()