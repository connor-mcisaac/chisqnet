#!/home/connor.mcisaac/envs/tensor-imbh/bin/python3.6

import argparse
import logging
import h5py
import numpy as np
from preprocessing import DataCollector, AttributeCollector


class DataCollector(object):

    def __init__(self):
        self.datasets = {}

    def __call__(self, name, node):
        if not isinstance(node, h5py.Dataset):
            pass
        elif name not in self.datasets.keys():
            self.datasets[name] = [node[:]]
        else:
            self.datasets[name].append(node[:])

    def concatenate_datasets(self):
        for k, v in self.datasets.items():
            self.datasets[k] = np.concatenate(v)

    def check_lengths(self):
        lengths = np.array([len(v) for v in self.datasets.values()])
        if len(np.unique(lengths)) > 1:
            raise ValueError("All datasets do not have the same length")


class AttributeCollector(object):

    def __init__(self):
        self.attrs = {}

    def __call__(self, group):
        for k in group.attrs.keys():
            if k not in self.attrs.keys():
                self.attrs[k] = group.attrs[k]
            elif self.attrs[k] != group.attrs[k]:
                raise ValueError("Groups have different attributes")


parser = argparse.ArgumentParser()

# Gather inputs from preprocessing step
parser.add_argument("--sample-files", nargs='+', required=True,
                    help="List of sample files")

# Gather output options
parser.add_argument("--output-file", required=True,
                    help="Output location")

# Gather additional options
parser.add_argument("--verbose", action='store_true')

args = parser.parse_args()

if args.verbose:
    log_level = logging.DEBUG
else:
    log_level = logging.WARNING
logging.basicConfig(format='%(asctime)s : %(message)s', level=log_level)

logging.info("Reading and combining sample files")

file_attrs = AttributeCollector()
file_data = DataCollector()
for fp in args.sample_files:
    with h5py.File(fp, 'r') as f:
        file_attrs(f)
        f.visititems(file_data)

file_data.concatenate_datasets()
file_data.check_lengths()

logging.info("Saving combined sample file")

with h5py.File(args.output_file, 'w') as f:
    
    for k, v in file_attrs.attrs.items():
        f.attrs[k] = v

    for k, v in file_data.datasets.items():
        _ = f.create_dataset(k, data=v)

logging.info("Done!")
