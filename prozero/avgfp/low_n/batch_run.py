#!/usr/bin/env python3
"""Run the twenty low-N avGFP trials reported in the manuscript."""
import argparse
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0', help='CUDA_VISIBLE_DEVICES value')
    parser.add_argument('--method', default='prozero')
    parser.add_argument('--runs', type=int, default=20)
    args = parser.parse_args()
    for run in range(1, args.runs + 1):
        subprocess.run(
            [sys.executable, 'train.py', '--method', args.method,
             '--idx', str(run), '--gpus', args.gpus],
            check=True)


if __name__ == '__main__':
    main()
