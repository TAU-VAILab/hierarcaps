from argparse import ArgumentParser
import os
from transformers import logging as transformers_logging
import time

from evaluation.qualitative_evaluator import QualitativeEvaluator


def get_opts():
    parser = ArgumentParser()
    parser.add_argument('--candidates_filename', type=str,
                        default='../data/hierarcaps_test_expanded.csv.gz')
    parser.add_argument('--weights', '-w', type=str)
    parser.add_argument('--logfile', '-l', type=str)
    parser.add_argument('--csvfile', '-c', type=str)
    parser.add_argument('--imgs_dir', '-i', type=str, default='images')
    parser.add_argument('--expname', '-e', type=str)
    parser.add_argument('--steps', '-s', type=int, default=50)
    return parser.parse_args()


def run_qual(args):
    QualitativeEvaluator(args).run()


def main():
    transformers_logging.set_verbosity_error()
    args = get_opts()

    if args.expname is None:
        args.expname = 'clip'
        print(f'No experiment named passed; defaulting to {args.expname}')
    else:
        print("Experiment name:", args.expname)

    if args.logfile is None:
        args.logfile = f'output/qual_logs/qual_log_{args.expname}.txt'
        print(f'No log filename passed; defaulting to {args.logfile}')
    else:
        print("Will log to:", args.logfile)
    if args.csvfile is None:
        args.csvfile = f'output/qual_logs/qual_log_{args.expname}.csv'
        print(f'No csv filename passed; defaulting to {args.csvfile}')
    else:
        print("Will log to:", args.csvfile)

    logdir = os.path.dirname(args.logfile)
    if logdir != "":
        os.makedirs(logdir, exist_ok=True)

    start = time.time()
    run_qual(args)
    end = time.time()

    seconds = end - start
    minutes = seconds / 60
    print()
    print(f"Qualitative eval finished; {minutes:.1f}m elapsed")


if __name__ == "__main__":
    main()
