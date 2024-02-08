import pandas as pd
import numpy as np

# reading the homography matrices
# check if the file exists
if os.path.exists('homographies.npz'):
    # Load the homography matrices
    with np.load('homographies.npz', allow_pickle=True) as data:
        homographies = [data['arr_%d' % i] for i in range(len(data.files))]
else:
    print("homographies file does not exist.")
    homographies = []

# reading the CSV file
df = pd.read_csv(f'{HOME}/drive/MyDrive/CSVOutputs/video_0044.mp4.csv')

# sorting the DataFrame
df = df.sort_values(by=['<frame>', '<id>'])

# initializing an empty DataFrame to store the results
results = pd.DataFrame(columns=['<frame>', '<id>', '<speed>'])

for id in df['<id>'].unique():
    id_df = df[df['<id>'] == id]
    for i in range(1, len(id_df)):
        # calculating the center of the bounding box for the previous frame
        x1, y1, w1, h1 = id_df.iloc[i-1][['<bb_left>', '<bb_top>', '<bb_width>', '<bb_height>']]
        x2, y2 = x1 + w1, y1 + h1
        center_box1 = np.array([(x1 + x2) / 2, (y1 + y2) / 2, 1])
        # center_box1 = ((x1 + x2) / 2, (y1 + y2) / 2)

        H = homographies[i]
        if H is not None:
            center_box1_hom = np.dot(H, center_box1)
            center_box1_hom /= center_box1_hom[2]  # converting from homogeneous to Cartesian coordinates
            center_box1 = (center_box1_hom[0], center_box1_hom[1])

        # calculating the center of the bounding box for current frame
        v_x1, v_y1, v_w1, v_h1 = id_df.iloc[i][['<bb_left>', '<bb_top>', '<bb_width>', '<bb_height>']]
        v_x2, v_y2 = v_x1 + v_w1, v_y1 + v_h1
        center_box2 = ((v_x1 + v_x2) / 2, (v_y1 + v_y2) / 2)

        # calculating the Euclidean distance between the two centers
        speed = np.sqrt((center_box2[0] - center_box1[0])**2 + (center_box2[1] - center_box1[1])**2)

        # storing the frame number, id, and calculated speed
        results = results.append({'<frame>': id_df.iloc[i]['<frame>'], '<id>': id, '<speed>': speed}, ignore_index=True)

# writing the results DataFrame to a new CSV file
results = results.sort_values(by=['<frame>'])
results.to_csv(f'{HOME}/speeds.csv', index=False)