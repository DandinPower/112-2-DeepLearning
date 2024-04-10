import matplotlib.pyplot as plt
import numpy as np

# Define the data
data = [35.1057, 26.5196, 22.6205, 22.5679, 21.4783, 20.5855, 20.8481, 20.9925, 20.4411, 20.6380, 19.8241, 19.7847, 19.2727, 19.2333, 19.4302, 19.1939, 18.9182, 19.0889, 18.9445, 19.0626, 19.0101, 19.0232, 18.8788, 18.8788, 18.8788]
data = list(map(lambda x: x / 100, data))

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