#!/home/connor.mcisaac/envs/tensor-imbh/bin/python3.6

import argparse
import logging
import h5py
import numpy as np
from ligo.segments import segment, segmentlist, segmentlistdict
from preprocessing import TriggerList, gather_segments, DataCollector, AttributeCollector
from strain import segmentlistdict_to_xml


parser = argparse.ArgumentParser()

# Gather inputs from preprocessing step
parser.add_argument("--sample-files", nargs='+', required=True,
                    help="List of sample files")
parser.add_argument("--injection-files", nargs='+', required=True,
                    help="List of injection files ")

# Gather output options
parser.add_argument("--output-sample-file", required=True,
                    help="Output location for sample file")
parser.add_argument("--output-injection-file", required=True,
                    help="Output location for injection file")
parser.add_argument("--output-segment-file", required=True,
                    help="Output location for segment file")

# Gather segment planning information
parser.add_argument("--segment-files", nargs='+', required=True,
                    help="List of segment files")
parser.add_argument("--segment-length", required=True, type=int,
                    help="Length of segments in seconds")
parser.add_argument("--start-pad", required=True, type=int,
                    help="Length of start pad in in seconds")
parser.add_argument("--end-pad", required=True, type=int,
                    help="Length of end pad in seconds")
parser.add_argument("--trigger-pad", required=True, type=int,
                    help="Length of padding around triggers in seconds")

# Gather additional options
parser.add_argument("--verbose", action='store_true')

args = parser.parse_args()

if args.verbose:
    log_level = logging.DEBUG
else:
    log_level = logging.WARNING
logging.basicConfig(format='%(asctime)s : %(message)s', level=log_level)

logging.info("Reading and combining sample files")

samples = TriggerList.read_from_hdf(args.sample_files[0])
for fp in args.sample_files[1:]:
    update = TriggerList.read_from_hdf(fp)
    samples.append(update)

logging.info("Total triggers:")
for k, v in samples.nums.items():
    logging.info("{0}: {1}".format(k, v))

logging.info("Saving combined sample file")

samples.write_to_hdf(args.output_sample_file)

logging.info("Reading and combining injection files")

file_attrs = AttributeCollector()
groups_data = {}
groups_attrs = {}
for fp in args.injection_files:
    with h5py.File(fp, 'r') as f:
        
        file_attrs(f)

        for ifo in f.keys():
            if ifo not in groups_data.keys():
                groups_data[ifo] = DataCollector()
                groups_attrs[ifo] = AttributeCollector()
            f[ifo].visititems(groups_data[ifo])
            groups_attrs[ifo](f[ifo])

for group in groups_data.values():
    group.concatenate_datasets()
    group.check_lengths()

logging.info("Saving combined injection file")

with h5py.File(args.output_injection_file, 'w') as f:
    
    for k, v in file_attrs.attrs.items():
        f.attrs[k] = v

    for ifo in groups_data.keys():
        g = f.create_group(ifo)
        
        for k, v in groups_attrs[ifo].attrs.items():
            g.attrs[k] = v

        for k, v in groups_data[ifo].datasets.items():
            _ = g.create_dataset(k, data=v)

logging.info("Planning segment boundaries")

science_segs = {ifo: gather_segments(args.segment_files, 'DATA_ANALYSED', ifo)
                for ifo in samples.ifos}
trigger_segs = {ifo: gather_segments(args.segment_files, 'TRIGGERS_GENERATED', ifo)
                for ifo in samples.ifos}

segs = segmentlistdict()
for ifo in samples.ifos:
    l_segs = []
    a_segs = []
    for sci_seg in science_segs[ifo]:
        start = sci_seg[0] + args.start_pad + args.trigger_pad
        end = sci_seg[1] - args.end_pad - args.trigger_pad

        if (end - start) < args.segment_length:
            continue

        triggers = samples.get_time_cut(ifo, start, end)
        if len(triggers) == 0:
            continue

        trigger_start = np.floor(np.min(triggers['end_time']))
        trigger_end = np.ceil(np.max(triggers['end_time']))

        seg = segment(start, end) & segment(trigger_start, trigger_end)

        num_segs = int(np.ceil(1. * (seg[1] - seg[0]) / args.segment_length))

        for i in range(num_segs):
            start = seg[0] + i * args.segment_length
            end  = start + args.segment_length
            analyse_start = start

            if end > seg[1]:
                end = seg[1]
                start = seg[1] - args.segment_length

            l_segs.append(segment(start, end))
            a_segs.append(segment(analyse_start, end))
            
    segs[ifo + ':LOAD'] = segmentlist(l_segs)
    segs[ifo + ':ANALYSE'] = segmentlist(a_segs)

logging.info("Saving segments")

segmentlistdict_to_xml(segs, args.output_segment_file)

samples.apply_segments([args.output_segment_file], 'ANALYSE')

logging.info("Triggers after applying ANALYSE segments:")
for k, v in samples.nums.items():
    logging.info("{0}: {1}".format(k, v))

logging.info("Done!")
