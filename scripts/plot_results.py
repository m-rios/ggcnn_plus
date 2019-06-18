import pylab as plt
import numpy as np
import argparse
import os

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('results', help='Path to result folder of specific model')

    args = parser.parse_args()

    sim_res_fn = os.path.join(args.results, 'results.txt')
    iou_res_fn = os.path.join(args.results, 'iou/iou.txt')
    iou2_res_fn = os.path.join(args.results, 'evaluation_output.txt')

    sim_epochs = []
    sim_values = []

    with open(sim_res_fn, 'r') as f:
        for line in f.readlines():
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

    iou_epochs = []
    iou_values = []
    iou2_values = []

    if os.path.exists(iou_res_fn):
        with open(iou_res_fn, 'r') as f:
            for line in f.readlines():
                fields = line.split(' ')
                if fields[0] != 'Epoch':
                    continue
                epoch = int(fields[1].replace(':',''))
                value = float(fields[2].replace('%', ''))
                iou_values.append(value)
                iou_epochs.append(epoch)
            iou_values = np.array(iou_values)
            iou_epochs = np.array(iou_epochs)
            sort_idx = np.argsort(iou_epochs)
            iou_epochs = iou_epochs[sort_idx]
            iou_values = iou_values[sort_idx]

    if os.path.exists(iou2_res_fn):
        with open(iou2_res_fn, 'r') as f:
            # Assumes only one line
            fields = f.read().split('\t')[1:-1]
            iou2_values = [float(x)/100 for x in fields]


    plt.figure(figsize=(10,5))
    plt.subplot(1, 2, 1)
    plt.plot(sim_epochs, sim_values, 'b-o')
    plt.title('Simulation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(iou_epochs, iou_values, 'r-o')
    plt.plot(range(1, len(iou2_values)+1), iou2_values, 'g-o')
    plt.legend(['Jacquard', 'Cornell'])
    plt.title('IOU > 0.4')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.suptitle('')

    plt.show()

