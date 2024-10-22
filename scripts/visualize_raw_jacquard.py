#!/usr/bin/env python
import argparse
from utils.dataset import Jacquard
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default=os.environ['JACQUARD_PATH'])
    parser.add_argument('--scene', default=None)
    args = parser.parse_args()

    jaq = Jacquard(args.dataset)

    if args.scene is None:
        for key in jaq.keys:
            jaq.plot(key)
    else:
        jaq.plot(args.scene)

