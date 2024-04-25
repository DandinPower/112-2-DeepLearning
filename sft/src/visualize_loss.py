import json
import matplotlib.pyplot as plt

from argparse import ArgumentParser, Namespace


def main(args: Namespace):

    # Open and load the JSON file
    with open(args.trainer_state_json_path) as f:
        data = json.load(f)

    # Extract the 'log_history' list
    log_history = data['log_history']

    # Initialize lists to hold the data
    steps = []
    losses = []
    learning_rates = []

    eval_steps = []
    eval_losses = []

    # Populate the lists with data
    for record in log_history:
        if record.get('loss'):
            steps.append(record['step'])
            losses.append(record['loss'])
            learning_rates.append(record['learning_rate'])
        elif record.get('eval_loss'):
            eval_steps.append(record['step'])
            eval_losses.append(record['eval_loss'])

    # Create a figure
    fig, ax1 = plt.subplots()

    # Plot 'step' vs 'loss'
    ax1.set_xlabel('step')
    ax1.set_ylabel('loss')
    line1, = ax1.plot(steps, losses, color='blue')
    line2, = ax1.plot(eval_steps, eval_losses, color='red')
    ax1.tick_params(axis='y')

    # Annotate last values of train loss and eval loss
    ax1.annotate(f'{losses[-1]:.2f}', (steps[-1], losses[-1]),
                 textcoords="offset points", xytext=(-10, -10), ha='center', color='blue')
    ax1.annotate(f'{eval_losses[-1]:.2f}', (eval_steps[-1], eval_losses[-1]),
                 textcoords="offset points", xytext=(-10, -10), ha='center', color='red')

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()

    # We already handled the x-label with ax1
    # Plot 'step' vs 'learning_rate'
    ax2.set_ylabel('learning_rate')
    line3, = ax2.plot(steps, learning_rates, color='green')
    ax2.tick_params(axis='y')

    # Create a legend for all lines in both plots
    plt.legend([line1, line2, line3], [
               'train_loss', 'eval_loss', 'learning_rate'])

    # Save the figure
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.savefig(args.image_path, dpi=300)
    plt.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--trainer_state_json_path", type=str, required=True)
    parser.add_argument("--image_path", type=str, required=True)
    args = parser.parse_args()
    main(args)
