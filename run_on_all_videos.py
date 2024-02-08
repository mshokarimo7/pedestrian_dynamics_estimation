JAAD_PATH = f"{HOME}/drive/MyDrive/JAAD/JAAD_clips"
video_files = [file for file in os.listdir(JAAD_PATH) if file.endswith(".mp4")]

for video_file in video_files:
    # clearing variables just in case we have trailing ones after each video iteration
    video_name = None
    SOURCE_VIDEO_PATH = None
    TARGET_VIDEO_PATH = None
    output_csv_file = None
    csv_file = None
    csv_writer = None
    csv_columns = None
    byte_tracker = None
    generator = None
    box_annotator = None
    video_info = None
    results = None
    detections = None
    mask = None
    tracks = None
    tracker_id = None
    track_id_to_object = None
    rearranged_tracks = None
    labels = None

    # creating proper paths
    SOURCE_VIDEO_PATH = os.path.join(JAAD_PATH, video_file)
    video_name = os.path.basename(SOURCE_VIDEO_PATH)
    print(video_name)
    TARGET_VIDEO_PATH = f"{HOME}/drive/MyDrive/TrackerOutputs/{video_name}" # video file path where the tracker outputs video
    output_csv_file = f"{HOME}/drive/MyDrive/CSVOutputs/{video_name}.csv" # CSV file path of the tracker outputs

    # defining the CSV column names
    csv_columns = [
        '<frame>',
        '<id>',
        '<bb_left>',
        '<bb_top>',
        '<bb_width>',
        '<bb_height>',
        '<conf>',
        '<x>',
        '<y>',
        '<z>'
    ]

    # opening the CSV file for writing
    csv_file = open(output_csv_file, 'w', newline='')
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)

    # writing the header row
    csv_writer.writeheader()

    # creating the BYTETracker variable for later runtime
    byte_tracker = BYTETracker(BYTETrackerArgs())
    generator = get_video_frames_generator(SOURCE_VIDEO_PATH)

    box_annotator = BoxAnnotator(color=ColorPalette(), thickness=1, text_thickness=1, text_scale=0.4)

    video_info = VideoInfo.from_video_path(SOURCE_VIDEO_PATH)

    # looping over the frames
    with VideoSink(TARGET_VIDEO_PATH, video_info) as sink:
      for frame in tqdm(generator, total=video_info.total_frames):
        results = model(frame)[0]
        """
        Creating a detections variable which finds the
        XY coordinates of objects,
        confidence in the detected object,
        class id of the object, in this snippet of code -> enumerated id from the YOLO models
        """
        detections = Detections(
            xyxy=results.boxes.xyxy.cpu().numpy(),
            confidence=results.boxes.conf.cpu().numpy(),
            class_id=results.boxes.cls.cpu().numpy().astype(int)
        )

        # filtering out detections with unwanted classes
        mask = np.array([class_id in CLASS_ID for class_id in detections.class_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        # using ByteTrack to track the objects, and storing the tracklets
        tracks = byte_tracker.update(
            output_results=detections2boxes(detections=detections),
            img_info=frame.shape,
            img_size=frame.shape
        )

        # matching the tracklet bounding boxes with the detections' bounding boxes
        tracker_id = match_detections_with_tracks(detections=detections, tracks=tracks)
        detections.tracker_id = np.array(tracker_id)

        # getting rid of detections that are not being tracked
        mask = np.array([tracker_id is not None for tracker_id in detections.tracker_id], dtype=bool)
        detections.filter(mask=mask, inplace=True)

        """
        recreating a new list of rearranged tracklets according to the tracker_id array
        creating a dictionary to map 'track_id' values to their corresponding objects
        """
        track_id_to_object = {track.track_id: track for track in tracks}

        # creating a list to store rearranged objects
        rearranged_tracks = []

        # iterating through the 'tracker_id' array
        for tracker_ids in tracker_id:
            # checking if theres a corresponding object in the dictionary
            if tracker_ids in track_id_to_object:
                # appending the object to the list of rearranged objects
                rearranged_tracks.append(track_id_to_object[tracker_ids])
                # removing the object from the dictionary to avoid duplicates
                del track_id_to_object[tracker_ids]
        # rearranged_tracks contains the objects sorted based on the 'tracker_id' array


        xyz = -1
        # iterating through rearranged_tracks and writing each STrack to the CSV file
        for track in rearranged_tracks:
          # extracting the values we want from the tracklet
          frame_csv = track.frame_id
          track_id = track.track_id
          # extracting the 'tlwh' instead of '_tlwh' makes the accuracy much better
          tlwh = track.tlwh
          score = track.score
          # extracting individual values for bb_left, bb_top, bb_width, and bb_height
          # tlwh -> top left width height
          bb_left, bb_top, bb_width, bb_height = tlwh[0], tlwh[1], tlwh[2], tlwh[3]
          # creating a dictionary to represent the row to write into our csv file
          row = {
              '<frame>': frame_csv,
              '<id>': track_id,
              '<bb_left>': bb_left,
              '<bb_top>': bb_top,
              '<bb_width>': bb_width,
              '<bb_height>': bb_height,
              '<conf>': score,
              '<x>': xyz,
              '<y>': xyz,
              '<z>': xyz
          }
          # writing the row to the CSV file
          csv_writer.writerow(row)

        labels = [
            f"#{tracker_id} {CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
            for _, confidence, class_id, tracker_id
            in detections
        ]
        rearranged_tracks.clear()
        frame = box_annotator.annotate(frame=frame, detections=detections, labels=labels)
        sink.write_frame(frame)

    csv_file.close()