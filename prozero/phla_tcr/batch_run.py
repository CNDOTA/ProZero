#!/usr/bin/env python3
"""Run the 27 peptide-HLA-TCR tasks reported in the manuscript."""
import argparse
import subprocess
import sys


TASKS = [
    ('ALIHHNTHL', 'CAVNSNSGYALNF', 'CASSQSETGDGYTF'),
    ('ALIHHNTHL', 'CAMHRDDKIIF', 'CASSLAVQRPSGNTIYF'),
    ('ALIHHNTHL', 'CVVSGVNVWGTYKYIF', 'CASSIESGSKQRNEQFF'),
    ('ALIHHNTHL', 'CAVSDLNSGGYQKVTF', 'CASSPRDRVHEQYF'),
    ('HMTEVVRHC', 'CAMSGLKEDSSYKLIF', 'CASSIQQGADTQYF'),
    ('HMTEVVRHC', 'CAFMGYSGAGSYQLTF', 'CAISELVTGDSPLHF'),
    ('HMTEVVRHC', 'CALDIYPHDMRF', 'CASSLDPGDTGELFF'),
    ('HMTEVVRHC', 'CVVQPGGYQKVTF', 'CASSEGLWQVGDEQYF'),
    ('LLGATCMFV', 'CAADSWGKLQF', 'CATSDSTGSYGYTF'),
    ('LLGATCMFV', 'CAVNPSNQFYF', 'CASRGPYHNEQFF'),
    ('LLGATCMFV', 'CVVSEEYTNAGKSTF', 'CASSLERLRVYSGYTF'),
    ('LLGATCMFV', 'CAMDSSYKLIF', 'CASSALAGGQADTQYF'),
    ('LLGATCMFV', 'CAAGGSYIPTF', 'CASSGTGGYSGANVLTF'),
    ('LLGATCMFV', 'CAVNDYKLSF', 'CASSWTGANYGYTF'),
    ('LLGATCMFV', 'CAVYSGGYNKLIF', 'CASSFVNTGELFF'),
    ('RLARLALVL', 'CASMYSGGGADGLTF', 'CASSFFSNTGELFF'),
    ('RLARLALVL', 'CASGGGADGLTF', 'CASSFLTDTQYF'),
    ('RLARLALVL', 'CSSGGGADGLTF', 'CASMDLAFKQYF'),
    ('RLARLALVL', 'CAYRSGSDGGSQGNLIF', 'CASSQVSGYEQYF'),
    ('RLARLALVL', 'CAVRDDYGQNFVF', 'CASSPQGDNEQFF'),
    ('RLARLALVL', 'CAVPDDAGNMLTF', 'CASSELPAGGTNEQFF'),
    ('RLARLALVL', 'CAGGGGADGLTF', 'CASSYMGPEAFF'),
    ('YLEPGPVTA', 'CAPGIAGGTSYGKLTF', 'CASSLAYSYEQYF'),
    ('YLEPGPVTA', 'CGTETNTGNQFYF', 'CASSLGRYNEQFF'),
    ('YLEPGPVTA', 'CAASTSGGTSYGKLTF', 'CASSLGSSYEQYF'),
    ('YLEPGPVTA', 'CAVLSSGGSNYKLTF', 'CASSFIGGTDTQYF'),
    ('YLEPGPVTA', 'CATDGDTPLVF', 'CASSIGGPYEQYF'),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus', default='0', help='CUDA_VISIBLE_DEVICES value')
    parser.add_argument('--method', default='prozero')
    parser.add_argument('--start-task', type=int, default=1)
    parser.add_argument('--end-task', type=int, default=len(TASKS))
    args = parser.parse_args()
    if not 1 <= args.start_task <= args.end_task <= len(TASKS):
        parser.error('task range must satisfy 1 <= start-task <= end-task <= 27')
    for task_number in range(args.start_task, args.end_task + 1):
        peptide, cdr3a, cdr3b = TASKS[task_number - 1]
        subprocess.run(
            [sys.executable, 'train.py', '--method', args.method,
             '--idx', '1', '--gpus', args.gpus,
             '--start_peptide', peptide,
             '--cdr3a', cdr3a, '--cdr3b', cdr3b],
            check=True)


if __name__ == '__main__':
    main()
