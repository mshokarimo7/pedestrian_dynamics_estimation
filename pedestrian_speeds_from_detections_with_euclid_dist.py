import pandas as pd

JAAD_PATH = f"{HOME}/drive/MyDrive/JAAD/JAAD_clips"
video_file = "video_0001.mp4"
SOURCE_VIDEO_PATH = os.path.join(JAAD_PATH, video_file)
video_name = os.path.basename(SOURCE_VIDEO_PATH)
print(video_name)
TARGET_VIDEO_PATH = f"{HOME}/output1.mp4"

# BYTETracker running
byte_tracker = BYTETracker(BYTETrackerArgs())
generator = get_video_frames_generator(SOURCE_VIDEO_PATH)

box_annotator = BoxAnnotator(color=ColorPalette.from_hex(['#04b09f']), thickness=1, text_thickness=1, text_scale=0.4)

video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

prev_detections = None
frame_num = 0
# initializing an empty DataFrame to store the results
df = pd.DataFrame(columns=['<frame>', '<id>', '<speed>'])

with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
    for frame in tqdm(generator, total=video_info.total_frames):
        results = model(frame)[0]
        """
        Creating a detections variable which finds the
        XY coordinates of objects,
        confidence in the detected object
        class id of the object, in this snipped of code -> enumerated id from the YOLO models
        """
        detections = Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int)
        )

        # filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # where we are actually using ByteTrack to track the objects
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )

        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)

        # filtering out detections without trackers --- THE TWO LINES BELOW SOLVED THE TRICK WITH
        # THE FRAMES NOT BEING ABLE TO RUN PAST 18 frames "IndexError: index 0 is out of bounds for axis 0 with size 0"
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # dictionary to store the bbox difference
        xyxy_diff_dict = {}

        #if prev_detections is not empty:
        if prev_detections is not None:
            # creating a dictionary mapping tracker_id to xyxy for prev_detections
            prev_dict = {detection[-1]: detection[0] for detection in prev_detections}

            # creating a dictionary mapping tracker_id to xyxy for detections
            current_dict = {detection[-1]: detection[0] for detection in detections}


            # for each tracker_id in current detections, if it also exists in prev_detections,
            # calculate the difference between the xyxy values
            for tracker_id in current_dict:
                if tracker_id in prev_dict:
                    # center of the bbox for the previous frame
                    x1, y1, w1, h1 = prev_dict[tracker_id]
                    x2, y2 = x1 + w1, y1 + h1
                    center_box1 = ((x1 + x2) / 2, (y1 + y2) / 2)

                    # center of the bbox for the current frame
                    x1_current, y1_current, w1_current, h1_current = current_dict[tracker_id]
                    x2_current, y2_current = x1_current + w1_current, y1_current + h1_current
                    center_box2 = ((x1_current + x2_current) / 2, (y1_current + y2_current) / 2)

                    # calculating the Euclidean distance between the two centers
                    xyxy_diff = np.sqrt((center_box2[0] - center_box1[0])**2 + (center_box2[1] - center_box1[1])**2)

                    xyxy_diff_dict[tracker_id] = round(xyxy_diff, 4)  # storing xyxy_diff in a dictionary, which is mapped to tracker_id

                    # storing the frame number, id, and calculated speed
                    df = df.append({'<frame>': frame_num, '<id>': tracker_id, '<speed>': xyxy_diff}, ignore_index=True)

        prev_detections = detections
        frame_num = frame_num + 1

        labels = [
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f} speed = {xyxy_diff_dict.get(tracker_id)}"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        sink.write_frame(frame)

# writing the results DataFrame to a new CSV file
df = df.sort_values(by=['<frame>'])
df.to_csv(f'{HOME}/speeds.csv', index=False)