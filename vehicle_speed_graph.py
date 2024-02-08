import matplotlib.pyplot as plt

frames = np.arange(len(speeds))

plt.plot(frames, speeds, label='Speed pixels/s', color='green')
plt.xlabel('Frame')
plt.ylabel('Speed')
plt.title('Ego-vehicle speed')
plt.xticks(np.arange(0, len(speeds)+5, 15))
plt.legend()
plt.show()