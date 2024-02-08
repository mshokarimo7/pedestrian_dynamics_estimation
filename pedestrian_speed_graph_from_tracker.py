import matplotlib.pyplot as plt

# reading the speeds.csv file
df = pd.read_csv(f'{HOME}/speeds.csv')

# getting the unique pedestrian ids
ids = df['<id>'].unique()

# creating a new figure
plt.figure()

plt.ylim(0, 30)

for id in ids:
    # skipping current iteration if the max speed for the current ID is above 30
    if df[df['<id>'] == id]['<speed>'].max() > 30:
        continue

    id_df = df[df['<id>'] == id]
    plt.plot(id_df['<frame>'], id_df['<speed>'], label=f'Pedestrian {id}')

    # plotting a horizontal line for max speed
    max_speed = id_df['<speed>'].max()
    plt.axhline(y=max_speed, color='r', linestyle='--')


# adding a legend, title, and labels
plt.legend()
plt.title('Speed of each pedestrian over time')
plt.xlabel('Frame')
plt.ylabel('Speed')

# calculating the minimum and maximum y-axis labels
y_min = 0
y_max = 25 + 5

# setting the y-ticks to be at every integer
plt.yticks(np.arange(y_min, y_max+1, 1.0))
plt.xticks(np.arange(0, df['<frame>'].max() + 1, 15))


# getting current y-ticks and their labels
locs, labels = plt.yticks()

# setting every 5th label to be visible, the rest invisible
for i, label in enumerate(labels):
    label.set_visible(i % 5 == 0)

plt.show()