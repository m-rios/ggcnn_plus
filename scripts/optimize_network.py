import core.network as net
import core.network_optimization as opt
import argparse
import os
import datetime

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model', help='path to model file')
    parser.add_argument('dataset', help='path to dataset file')
    parser.add_argument('--epochs',default=10, type=int, help='Number of epochs to train children nodes for')
    parser.add_argument('--retrain_epochs',default=0, type=int, help='Number of epochs to train best node for')
    parser.add_argument('--width',default=3, type=int, help='Width of the beam search')
    parser.add_argument('--depth',default=5, type=int, help='Depth of the beam search')
    parser.add_argument('--results',default=os.environ['RESULTS_PATH'], help='Path where results directory will be created')
    parser.add_argument('--expand_transpose',action='store_true', help='If set it also explores Conv2DTranspose layers')

    args = parser.parse_args()

    dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    results_path = os.path.join(args.results, 'beam_search_' + dt)
    os.makedirs(results_path)

    network = net.Network(model_fn=args.model)
    optimizer = opt.NetworkOptimization('loss', args.dataset, debug=True, epochs=args.epochs, results_path=results_path, retrain_epochs=args.retrain_epochs, expand_transpose=args.expand_transpose)
    [nodes, scores, actions] = optimizer.run(network, k=args.width, depth=args.depth, minimize=True)

    results_fn = open(os.path.join(results_path, 'results.txt'), 'a')
    for node_idx, node in enumerate(nodes):
        results_fn.write('Node: {}\tScore: {}\tAction:{}\n'.format(node, scores[node_idx], actions[node_idx]))
