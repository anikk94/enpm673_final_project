import numpy as np
import cv2
import argparse
from math import sqrt, pow

def dist(x1, y1, x2, y2):
    return sqrt(pow(x1-x2, 2) + pow(y1-y2, 2))

def get_speed(ox, oy, nx, ny, lane=0):
    # print(f"measuring speed between {measurement_region[lane-1][0][1]} and {measurement_region[lane-1][1][1]}")
    if oy > measurement_region[lane][0][1] and oy < measurement_region[lane][1][1] and ox < 830 and ox > 770:
        pixeldistance = dist(ox, oy, nx, ny)
        pixelspersecond = pixeldistance * fps
        meterspersecond = pixelspersecond / pixelspermeter
        # return kmph
        return round(meterspersecond * 3.6, 1)
    return None

# video_file = "highway1.mp4"
video_file = "highway2.mp4"
# video_file = "highway3.mp4"
# video_file = "highway4.mp4"
# video_file = "highway5.mp4"

# region1= np.array([[770,  23], [770, 108], [838, 108], [838,  23]])
# region2= np.array([[770, 160], [770, 209], [838, 209], [838, 160]])
# region3= np.array([[770, 218], [770, 260], [838, 260], [838, 218]])
# region4= np.array([[770, 268], [770, 310], [838, 310], [838, 268]])
# region5= np.array([[770, 329], [770, 370], [838, 370], [838, 329]])
# region6= np.array([[770, 379], [770, 418], [838, 418], [838, 379]])
# region7= np.array([[770, 427], [770, 480], [838, 480], [838, 427]])
# region8= np.array([[770, 544], [770, 611], [838, 611], [838, 544]])
# region9= np.array([[770, 629], [770, 705], [838, 705], [838, 629]])

measurement_region = [
    np.array([[770,  23], [770, 108], [838, 108], [838,  23]]),
    np.array([[770, 160], [770, 209], [838, 209], [838, 160]]),
    np.array([[770, 218], [770, 260], [838, 260], [838, 218]]),
    np.array([[770, 268], [770, 310], [838, 310], [838, 268]]),
    np.array([[770, 329], [770, 370], [838, 370], [838, 329]]),
    np.array([[770, 379], [770, 418], [838, 418], [838, 379]]),
    np.array([[770, 427], [770, 480], [838, 480], [838, 427]]),
    np.array([[770, 544], [770, 611], [838, 611], [838, 544]]),
    np.array([[770, 629], [770, 705], [838, 705], [838, 629]])
    ]

lanespeed = [0, 0, 0, 0, 0, 0, 0, 0, 0]

cap = cv2.VideoCapture(video_file)

# Constants
lanemarkingpixel = 768-694
pixelspermeter = lanemarkingpixel / 3.048
fps = round(cap.get(cv2.CAP_PROP_FPS))
print(f"FPS: {fps}")
# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.01,
                       minDistance = 20,
                       blockSize = 3 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
# Take first frame and find corners in it
ret, old_frame = cap.read()
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# check corners
# for i in p0:
#     x, y = i.ravel()
#     old_frame = cv2.circle(old_frame, (int(x), int(y)), 5, (0, 0, 255), -1)
# cv2.imshow("old_frame",old_frame)
# cv2.waitKey(0)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

feature_refresh_counter = 0

while(1):
    ret, frame = cap.read()
    if not ret:
        print('No frames grabbed!')
        break

    # speed cam regions
    cv2.fillPoly(mask, [measurement_region[0]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[1]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[2]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[3]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[4]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[5]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[6]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[7]], (0,0,50))
    cv2.fillPoly(mask, [measurement_region[8]], (0,0,50))

    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # recalculate good features
    if feature_refresh_counter % 20 == 0:
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        # mask = np.zeros_like(old_frame)


    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Select good points
    if p1 is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
    
    # draw the tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # # region2
        # if b > 160 and b < 209 and a < 830 and a > 770:
        # region6
        for i in range(9):
            speed = get_speed(a, b, c, d, lane=i)
            if speed:
                lanespeed[i] = speed

        print("{:4} | {:4} | {:4} | {:4} | {:4} | {:4} | {:4} | {:4} | {:4}"
          .format(lanespeed[0], lanespeed[1], lanespeed[2], lanespeed[3], 
            lanespeed[4], lanespeed[5], lanespeed[6], lanespeed[7], lanespeed[8]))

        # mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        # mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0,255,0), 2)
        frame = cv2.circle(frame, (int(a), int(b)), 5, (0,255,0), -1)
        frame = cv2.circle(frame, (int(c), int(d)), 5, (0,0,255), -1)
    img = cv2.add(frame, mask)
    # img = frame
    cv2.imshow('frame', img)

    key = cv2.waitKey(fps)
    if key == ord("q"):
        break
    
    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)
    feature_refresh_counter += 1
cv2.destroyAllWindows()