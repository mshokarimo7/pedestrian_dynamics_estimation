# setting the base path
base_path = f'{HOME}/drive/MyDrive/TrackEval/data/gt/mot_challenge/JAAD-train'
JAAD_PATH = f"{HOME}/drive/MyDrive/JAAD/JAAD_clips"

# creating folders and files
for i in range(1, 347):
    video_folder = os.path.join(base_path, f'video_{i:04d}')
    gt_folder = os.path.join(video_folder, 'gt')
    seqinfo_file = os.path.join(video_folder, 'seqinfo.ini')

    # creating the video folders
    os.makedirs(video_folder, exist_ok=True)

    # creating the gt folders
    os.makedirs(gt_folder, exist_ok=True)

    # getting video file path
    video_file = os.path.join(JAAD_PATH, f'video_{i:04d}.mp4')


    # extracting the video info
    cap = cv2.VideoCapture(video_file)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    seq_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    im_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    im_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    # creating seqinfo.ini files and populating them
    with open(seqinfo_file, 'w') as seqinfo:
        seqinfo.write('[Sequence]\n')
        seqinfo.write(f'name=video_{i:04d}\n')
        seqinfo.write('imDir=img1\n')
        seqinfo.write(f'frameRate={frame_rate}\n')
        seqinfo.write(f'seqLength={seq_length}\n')
        seqinfo.write(f'imWidth={im_width}\n')
        seqinfo.write(f'imHeight={im_height}\n')
        seqinfo.write('imExt=.jpg\n')


import os
import shutil

# source directory containing the text format CSV files
source_directory = f'{HOME}/drive/MyDrive/JAAD/Conversion/TextFormat'

# destination directory of ground truth
destination_directory = f'{HOME}/drive/MyDrive/TrackEval/data/gt/mot_challenge/JAAD-train'

# looping through all subfolders in the destination directory
for subfolder in os.listdir(destination_directory):
    subfolder_path = os.path.join(destination_directory, subfolder)

    # checking if the subfolder has a 'gt' folder
    gt_folder_path = os.path.join(subfolder_path, 'gt')
    if os.path.exists(gt_folder_path) and os.path.isdir(gt_folder_path):
        # creating the source file path based on the subfolder name
        source_file_path = os.path.join(source_directory, f"{subfolder}.txt")

        # creating the destination file path
        destination_file_path = os.path.join(gt_folder_path, 'gt.txt')

        # checking if the source file exists
        if os.path.exists(source_file_path):
            # copying the ground truth .txt file to the destination folder
            shutil.copy(source_file_path, destination_file_path)
            print(f"copied {source_file_path} to {destination_file_path}")
        else:
            print(f"source file {source_file_path} not found for {subfolder}")
    else:
        print(f"no 'gt' folder found for {subfolder}")
