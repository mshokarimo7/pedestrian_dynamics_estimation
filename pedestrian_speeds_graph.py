import matplotlib.pyplot as plt

# reading the speeds.csv file
df = pd.read_csv(f'{HOME}/speeds.csv')

# getting the unique pedestrian ids
ids = df['<id>'].unique()

# creating a new figure
plt.figure()

# for each id, plotting the speed over time
for id in ids:
    id_df = df[df['<id>'] == id]
    plt.plot(id_df['<frame>'], id_df['<speed>'], label=f'Pedestrian {id}')

# adding a legend, title, and labels
plt.legend()
plt.title('Speed of each pedestrian over time')
plt.xlabel('Frame')
plt.ylabel('Speed')

plt.show()