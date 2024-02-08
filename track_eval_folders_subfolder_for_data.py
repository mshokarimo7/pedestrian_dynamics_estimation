import shutil
# setting the source and destination folders
source_folder = f'{HOME}/drive/MyDrive/CSVOutputs/TxtOutputs'
destination_folder = f'{HOME}/drive/MyDrive/TrackEval/data/trackers/mot_challenge/JAAD-train/BYTETrack/data'

# iterating over each file in the source folder
for filename in os.listdir(source_folder):
    if filename.endswith('.txt'):
        source_file = os.path.join(source_folder, filename)
        destination_file = os.path.join(destination_folder, filename)

        # copying the file to the destination folder
        shutil.copyfile(source_file, destination_file)