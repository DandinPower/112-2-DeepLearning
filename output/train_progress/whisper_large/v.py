import matplotlib.pyplot as plt
import numpy as np

# Define the data
with open('whisper-large-tawiwanese-asr_wer_progress.txt', 'r') as f:
    data = f.readlines()
    data = [size.split('\n')[0] for size in data]
    data = list(map(lambda x: float(x), data))

# Define the epochs
epochs = np.arange(1, len(data) + 1)

# Create the plot
plt.plot(epochs, data)

# Label the axes and give the plot a title
plt.xlabel('Epoch')
plt.ylabel('WER')
plt.title('WER Progress')

# Display the plot
plt.savefig('WER.png')