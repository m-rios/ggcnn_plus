import pylab as plt
import numpy as np
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='Path to results.txt file')
    parser.add_argument('title', help='Title of the figure')
    parser.add_argument('--sim_file', help='Path to results.txt file')

    args = parser.parse_args()

    with open(args.file, 'r') as f:
        nodes, scores, actions = zip(*[line.split('\t') for line in f if 'Node:' in line])

    nodes = [node.replace('Node:', '') for node in nodes]
    scores = [float(score.replace('Score:', '')) for score in scores]
    actions = [action.replace('Action:', '') for action in actions]

    if args.sim_file is not None:
        sim_epochs = []
        sim_values = []

        with open(args.sim_file, 'r') as f:
            for line in [l for l in f.readlines() if 'Epoch' in l]:
                fields = line.split(' ')
                epoch = int(fields[1].replace(':',''))
                value = float(fields[2].replace('%', ''))
                sim_values.append(value)
                sim_epochs.append(epoch)
            sim_values = np.array(sim_values)
            sim_epochs = np.array(sim_epochs)
            sort_idx = np.argsort(sim_epochs)
            sim_epochs = sim_epochs[sort_idx]
            sim_values = sim_values[sort_idx]

    plt.plot(range(len(scores)), scores, label='IOU > .25')
    if args.sim_file is not None:
        plt.plot(sim_epochs, sim_values, label='Simulation')
    plt.title(args.title)
    plt.xlabel('Depth')
    plt.ylabel('Success rate')
    plt.legend()
    plt.savefig(args.title)
