import json
import matplotlib.pyplot as plt

# Load the trainer_state.json file
with open('minicpm_trainer_state.json') as f:
    data = json.load(f)

# Extract the log history
log_history = data['log_history']

# Extract steps and losses
steps = []
losses = []

for log in log_history:
    if log.get('loss') is not None:
        steps.append(log['step'])
        losses.append(log['loss'])

# Plot the loss progress
plt.figure(figsize=(10, 6))
plt.plot(steps, losses, marker='o', linestyle='-')
plt.title('Trainer Loss Progress')
plt.xlabel('Step')
plt.ylabel('Loss(log scale)')
plt.yscale('log')
plt.grid(True)
plt.savefig('minicpm_loss.png', dpi=300)
