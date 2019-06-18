import os
import glob
import argparse
from subprocess import Popen
import time

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result', help='Path to epoch folder')
    parser.add_argument('--fails', action='store_true', help='Show failures (successes by default)')

    args = parser.parse_args()

    failure_fn = args.result + '/sim_logs/failures.txt'

    output_path = args.result + '/output'
    log_path = args.result + '/sim_logs'

    vid_p = None
    img_p = None

    with open(failure_fn) as f:
        failures = set([x.replace('\n', '') for x in f.readlines()])
        results = list(failures)

    if not args.fails:
        scenes = set([x.split('/')[-1].replace('.png', '') for x in glob.glob(output_path + '/*.png')])
        successes = scenes - failures
        results = list(successes)

    for result in results:
        result_img = output_path + '/{}.png'.format(result)
        result_vid = log_path + '/{}.mp4'.format(result)
        vid_p = Popen('mpv --loop-file {}'.format(result_vid), shell=True)
        img_p = Popen('feh {}'.format(result_img), shell=True)
        vid_p.wait()
        img_p.wait()




