import cv2
import numpy as np
cv2.namedWindow("Camera", cv2.WINDOW_KEEPRATIO)

cam = cv2.VideoCapture(0)
cam.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

roi = None
imCrop = None
while cam.isOpened():
    ret, frame = cam.read()

    if roi is None:
        roi = cv2.selectROI("Camera", frame)
        imCrop = frame[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        
    if roi is not None:
        orb = cv2.SIFT_create()
        key_points1, descriptors1 = orb.detectAndCompute(imCrop, None)
        key_points2, descriptors2 = orb.detectAndCompute(frame, None)
        matcher = cv2.BFMatcher()
        matches = matcher.knnMatch(descriptors1, descriptors2, k=2)

        best = []

        for m1, m2 in matches:
            if m1.distance < 0.75 * m2.distance:
                best.append([m1])


        print(f"All matches {len(matches)}, best matches {len(best)}")

        if len(best) > 30:
            src_pts = np.float32([key_points1[m[0].queryIdx].pt for m in best]).reshape(-1, 1, 2)
            dst_pts = np.float32([key_points2[m[0].trainIdx].pt for m in best]).reshape(-1, 1, 2)
            M, hmask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            print(imCrop.shape)
            h, w, c = np.array(imCrop).shape
            pts = np.float32([[0, 0], [0, h-1], [w- 1, h-1], [w - 1, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, M)
            result = cv2.polylines(frame, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
        else:
            print("Not found") 
            mask = None

        matches_image = cv2.drawMatchesKnn(imCrop, key_points1, frame, key_points2, best, frame)
        cv2.imshow("Camera", frame)
        
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    
        

cam.release()
cv2.destroyAllWindows()
