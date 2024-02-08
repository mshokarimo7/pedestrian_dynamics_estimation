import cv2
import numpy as np

def create_optical_flow_visualization_with_mask_simple(frame, yolo_mask, darken_factor=0.7):
    # converting YOLO mask to RGB for overlay
    yolo_mask_rgb = cv2.cvtColor(yolo_mask, cv2.COLOR_GRAY2BGR)

    inverted_mask = cv2.bitwise_not(yolo_mask)
    darkened_region = cv2.multiply(frame, yolo_mask_rgb, scale=darken_factor)
    result = cv2.bitwise_or(darkened_region, frame, mask=inverted_mask)

    return result

# Lukas-Kanade optical flow parameters
LK_PARAMS = dict(winSize=(35, 35), maxLevel=3, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.02))

def calculate_trajectory(video_path, output_path, max_corners=100, quality_level=0.3, min_distance=7, block_size=7, ransac_threshold=1.0):

    cap = cv2.VideoCapture(video_path)

    # reading the first frame
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # running YOLO -------------------------------------------------------------
    results = model(old_frame)[0]
    detections = Detections(
        xyxy=results.boxes.xyxy.cpu().numpy(),
        confidence=results.boxes.conf.cpu().numpy(),
        class_id=results.boxes.cls.cpu().numpy().astype(int)
    )
    # getting the class IDs from the detections
    class_ids = detections.class_id

    # getting the indices of detections where the class ID is 0 (person)
    person_indices = np.where(class_ids == 0)[0]

    # filtering the bounding boxes to include only 'person' class
    person_bboxes = detections.xyxy[person_indices]

    yolo_mask = np.ones(old_frame.shape[:2], dtype=np.uint8) * 255

    # setting the corresponding region in the mask to 0 for each persons bbox
    for bbox in person_bboxes:
        x1, y1, x2, y2 = map(int, bbox)
        yolo_mask[y1:y2, x1:x2] = 0
    # --------------------------------------------------------------------------

    # initializing the keypoints for the first frame
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=yolo_mask, maxCorners=max_corners, qualityLevel=quality_level,
                                 minDistance=min_distance, blockSize=block_size)
    speeds = []

    # creating a VideoWriter object 'out' to save the frames as a video file
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

    # creating the homography matrix list and setting the first element to None
    homographies = [None]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # running YOLO Again ---------------------------------------------------
        results1 = model(frame)[0]
        detections1 = Detections(
            xyxy=results1.boxes.xyxy.cpu().numpy(),
            confidence=results1.boxes.conf.cpu().numpy(),
            class_id=results1.boxes.cls.cpu().numpy().astype(int)
        )
        # getting the class IDs from the detections
        class_ids1 = detections1.class_id

        # getting the indices of detections where the class ID is 0 (person)
        person_indices1 = np.where(class_ids1 == 0)[0]

        # filtering the bounding boxes to include only 'person' class
        person_bboxes1 = detections1.xyxy[person_indices1]

        yolo_mask1 = np.ones(frame.shape[:2], dtype=np.uint8) * 255

        # setting the corresponding region in the mask to 0 for each persons bbox
        for bbox1 in person_bboxes1:
            x1_, y1_, x2_, y2_ = map(int, bbox1)
            yolo_mask1[y1_:y2_, x1_:x2_] = 0
        # ----------------------------------------------------------------------

        # calculating the optical flow using Lukas-Kanade method
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **LK_PARAMS)

        # creating the optical flow visualization
        vis = create_optical_flow_visualization_with_mask_simple(frame, yolo_mask1)
        # vis = create_optical_flow_visualization(frame, p0, p1, threshold=0.6, yolo_mask=yolo_mask1)
        overlay = cv2.add(vis, frame)
        # writing the visualization frame to the output video
        out.write(overlay)

        # selecting good points
        good_new = p1[st == 1]
        good_old = p0[st == 1]

        # using RANSAC to further refine the set of feature points
        H, mask = cv2.findHomography(good_old, good_new, cv2.RANSAC, ransac_threshold)
        # storing the homography matrix
        homographies.append(H)
        # applying the mask to only keep the inliers
        good_new = good_new[mask.ravel() == 1]
        good_old = good_old[mask.ravel() == 1]

        # calculating the speed between frames using inliers
        if len(good_new) > 0:
            speeds.append(np.linalg.norm(good_new.mean(axis=0) - good_old.mean(axis=0)))

        # updating previous frame and points
        old_gray = frame_gray.copy()
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=yolo_mask1, maxCorners=max_corners, qualityLevel=quality_level,
                                     minDistance=min_distance, blockSize=block_size)

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    np.savez('homographies.npz', *homographies)
    return np.array(speeds)

video_path = f'{HOME}/drive/MyDrive/JAAD/JAAD_clips/video_0001.mp4'
output_path = f'{HOME}/output.mp4'

speeds = calculate_trajectory(video_path, output_path)

print(speeds)
