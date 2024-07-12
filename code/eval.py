from argparse import ArgumentParser
import os
import time
import json
from transformers import logging as transformers_logging

from evaluation.hierarchical_retrieval import HCEvaluator


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--test_csv_filename', type=str, default='../data/hierarcaps_test.csv')
    parser.add_argument('--test_images_dir', type=str, default='../assets/test_imgs')
    parser.add_argument('--weights', '-w', type=str)
    parser.add_argument('--expname', '-e', type=str)
    parser.add_argument('--logfile', '-l', type=str)
    parser.add_argument('--steps', '-s', type=int, default=50)
    parser.add_argument('-bc', '--base_checkpoint',
                        default='openai/clip-vit-base-patch32', type=str)
    return parser.parse_args()


def run(args, log):
    e = HCEvaluator(args, 'hierarcaps', log)
    e.run()


def main():
    transformers_logging.set_verbosity_error()
    args = get_opts()

    if args.expname is None:
        args.expname = f'clip_eval'
        print(f'No experiment named passed; defaulting to {args.expname}')
    else:
        print("Experiment name:", args.expname)

    if args.logfile is None:
        args.logfile = f'output/eval_logs/eval_log_{args.expname}.json'
        print(f'No log filename passed; defaulting to {args.logfile}')
    else:
        print("Will log to:", args.logfile)

    logdir = os.path.dirname(args.logfile)
    if logdir != "":
        os.makedirs(logdir, exist_ok=True)

    print("Starting evaluation")
    start = time.time()

    log = {}
    run(args, log)

    if args.logfile is not None:
        print("Saving log to:", args.logfile)
        with open(args.logfile, 'w') as f:
            json.dump(log, f, indent=4)
        print("Log saved")

    end = time.time()

    seconds = end - start
    minutes = seconds / 60
    print()
    print(f"Evaluation finished; {minutes:.1f}m elapsed")


if __name__ == "__main__":
    main()
