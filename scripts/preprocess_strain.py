#!/home/connor.mcisaac/envs/tensor-imbh/bin/python3.6

import argparse
import logging, time
import h5py
import numpy as np
from preprocessing import TriggerList, add_params
from pycbc.workflow import SegFile
from strain import xml_to_segmentlistdict
from strain import insert_strain_option_group, strain_from_cli, StrainSegmentsCut
from pycbc import psd, waveform, DYN_RANGE_FAC
from pycbc.types import zeros, float32, complex64


parser = argparse.ArgumentParser()

# Gather inputs from planning step, Injection file is included in strain options group
parser.add_argument("--sample-file", required=True,
                    help="Sample file to be used in training")
parser.add_argument("--segment-file", required=True,
                    help="Segment file to be used in training")

# Gather arguments for loading and preparing strain data
insert_strain_option_group(parser)
StrainSegmentsCut.insert_segment_option_group(parser)
psd.insert_psd_option_group(parser)
parser.add_argument("--low-frequency-cutoff", type=float,
                  help="The low frequency cutoff to use for filtering (Hz)")
parser.add_argument("--snr-cut-width", default=0.1, type=float,
                    help="The width around each trigger to cut from the SNR timeseries")

# Gather output options
parser.add_argument("--output-file", required=True,
                    help="Name of the output file")

# Gather additional options
parser.add_argument("--num-jobs", required=True, type=int,
                    help="The number of jobs used to preprocess all samples")
parser.add_argument("--job-num", required=True, type=int,
                    help="The job number, zero indexed")
parser.add_argument("--chunk-num", default=1, type=int,
                    help="The number of samples to hold in a hdf chunk")
parser.add_argument("--extend-num", default=1, type=int,
                    help="The number of samples to load before extending the file")
parser.add_argument("--verbose", action='store_true')

args = parser.parse_args()

if args.verbose:
    log_level = logging.DEBUG
else:
    log_level = logging.WARNING
logging.basicConfig(format='%(asctime)s : %(message)s', level=log_level)

# Check option groups
StrainSegmentsCut.verify_segment_options(args, parser)
psd.verify_psd_options(args, parser)

if 0 > args.job_num or args.job_num >= args.num_jobs:
    raise ValueError("--job-num must be between 0 and --num-jobs")

samples = TriggerList.read_from_hdf(args.sample_file)
segdict = xml_to_segmentlistdict(args.segment_file)

segments_load = []
segments_ana = []
segments_ifo = []
for ifo in samples.ifos:
    if ifo + ':LOAD' in segdict.keys() and ifo + ':ANALYSE' in segdict.keys():
        segments_load += list(segdict[ifo + ':LOAD'])
        segments_ana += list(segdict[ifo + ':ANALYSE'])
        segments_ifo += [ifo] * len(segdict[ifo + ':LOAD'])

seg_num = len(segments_load)

order = np.arange(seg_num)
div = seg_num // args.num_jobs
if args.job_num < args.num_jobs:
    start = args.job_num * div
    end = start + div
    idxs = order[start:end]
else:
    start = args.job_num * div
    idxs = order[start:]

logging.info("Each job preprocessing ~ {0} segments".format(div))

delta_f = 1. / args.segment_length
tlen = int(args.segment_length * args.sample_rate)
flen = tlen // 2 + 1

with h5py.File(args.output_file, 'w') as f:

    stildes = []
    psds = []
    cuts = []
    samps = []

    for idx in idxs:

        logging.info("Processing segment {0}".format(idx))

        seg_load = segments_load[idx]
        seg_ana = segments_ana[idx]
        seg_ifo = segments_ifo[idx]

        seg_samples = samples.get_time_cut(seg_ifo, seg_ana[0], seg_ana[1])
        seg_times = seg_samples['end_time']
        trig_num = len(seg_samples)

        seg_samples = add_params(seg_samples, np.array([[seg_ifo] * trig_num], dtype='U2'),
                                 ['ifo'], ['U2'])

        logging.info("Loading strain")
        seg_strain = strain_from_cli(seg_ifo, seg_load, args,
                                     dyn_range_fac=DYN_RANGE_FAC, precision='single')
        logging.info("Segmenting strain")
        seg_segs = StrainSegmentsCut.from_cli(args, seg_strain)
        seg_segs.cut(seg_times, args.snr_cut_width)

        logging.info("Fourier transforming strain")
        seg_stilde = seg_segs.fourier_segments()

        logging.info("Calculating PSDs")
        psd.associate_psds_to_segments(args, seg_stilde, seg_strain,
                                       seg_segs.freq_len, seg_segs.delta_f,
                                       args.low_frequency_cutoff,
                                       dyn_range_factor=DYN_RANGE_FAC, precision='single')

        seg_psds = [seg.psd for seg in seg_stilde]

        logging.info("Selecting sample data")
        for i in range(trig_num):

            sidx = seg_segs.sample_segments[i]
            cidx = seg_segs.sample_slices[i]

            stildes.append(seg_stilde[sidx].numpy())
            psds.append(seg_psds[sidx].numpy())
            cuts.append(np.array([cidx.start, cidx.stop]))
            samps.append(seg_samples[i])

        if len(stildes) > args.extend_num:
            logging.info("Writing to file")
            if "stilde" not in f.keys():
                stilde_group = f.create_dataset("stilde", (len(stildes), flen), dtype='c8',
                                                chunks=(args.chunk_num, flen), maxshape=(None, flen))
                psd_group = f.create_dataset("psd", (len(stildes), flen), dtype='f4',
                                             chunks=(args.chunk_num, flen), maxshape=(None, flen))
            else:
                stilde_group.resize(stilde_group.len() + len(stildes), axis=0)
                psd_group.resize(psd_group.len() + len(psds), axis=0)

            stilde_group[-len(stildes):, :] = np.stack(stildes)
            psd_group[-len(psds):, :] = np.stack(psds)

            stildes = []
            psds = []

    if len(stildes) > 0:
        logging.info("Writing to file")
        stilde_group.resize(stilde_group.len() + len(stildes), axis=0)
        psd_group.resize(psd_group.len() + len(psds), axis=0)

        stilde_group[-len(stildes):, :] = np.stack(stildes)
        psd_group[-len(psds):, :] = np.stack(psds)

    logging.info("Writing cuts and samples")
    cuts = np.stack(cuts)
    samps = np.stack(samps)

    cut_group = f.create_dataset("cut", data=cuts)
    
    for p, d in zip(samples._params, samples._dtypes):
        data = samps[p][:]
        if d.startswith('U'):
            data = data.astype('S')
        _ = f.create_dataset(p, data=data)
    
    _ = f.create_dataset('ifo', data=samps['ifo'][:].astype('S'))
    f.attrs['flow'] = args.low_frequency_cutoff
    f.attrs['flen'] = flen
    f.attrs['delta_f'] = delta_f
    f.attrs['tlen'] = tlen
    f.attrs['sample_rate'] = args.sample_rate

logging.info("Done")
