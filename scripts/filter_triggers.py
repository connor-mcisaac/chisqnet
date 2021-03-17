#!/home/connor.mcisaac/envs/tensor-imbh/bin/python3.6

import argparse
import logging
from preprocessing import TriggerList, InjectionTriggers


parser = argparse.ArgumentParser()

# Gather inputs from search workflow used to select data for training
parser.add_argument("--trigger-files", nargs='+', required=True,
                    help="List of single detector trigger files used to "
                         "select sample times for training")
parser.add_argument("--inj-find-files", nargs='+', required=True,
                    help="List of injfind files used to select observable "
                         "injections for use in training")
parser.add_argument("--inj-trigger-files", nargs='+', required=True,
                    help="List of lists of single detector trigger files "
                         "associated to each of the injfind files. "
                         "For each injfind file give a list of the form "
                         "H1-file.hdf,L1-file.hdf,V1-file.hdf")
parser.add_argument("--inj-approximants", nargs='+', required=True,
                    help="The approximant associated to each of the "
                         "injfind files")
parser.add_argument("--inj-f-lowers", nargs='+', type=float,
                    help="The f-lower associated to each of the "
                         "injfind files")
parser.add_argument("--segment-files", nargs='+', required=True,
                    help='Segment files containing "DATA_ANALYSED" and '
                         '"TRIGGERS_GENERATED" segments')
parser.add_argument("--veto-files", nargs='+', required=True,
                    help='Segment file conatining the "closed_box" veto '
                         'used to remove foreground triggers')
parser.add_argument("--bank-file", required=True,
                    help="Template bank file used in search workflow")

# Gather output options
parser.add_argument("--output-sample-file", required=True,
                    help="Output location for training sample times")
parser.add_argument("--output-injection-file", required=True,
                    help="Output location for injection file")

# Gather inputs for filtering triggers
parser.add_argument("--trigger-snr-cut", type=float, default=0,
                    help="Threshold for removing triggers below snr")
parser.add_argument("--trigger-newsnr-cut", type=float, default=0,
                    help="Threshold for removing triggers below newsnr")
parser.add_argument("--trigger-newsnr-sg-cut", type=float, default=0,
                    help="Threshold for removing triggers below newsnr_sg")
parser.add_argument("--trigger-cluster-param",
                    help="Parameter to use when clustering triggers")
parser.add_argument("--trigger-cluster-window", type=float,
                    help="Window in seconds to use when clustering triggers")

# Gather inputs for filtering injections
parser.add_argument("--injection-snr-cut", type=float, default=0,
                    help="Threshold for removing injections below snr")
parser.add_argument("--injection-newsnr-cut", type=float, default=0,
                    help="Threshold for removing injections below newsnr")
parser.add_argument("--injection-newsnr-sg-cut", type=float, default=0,
                    help="Threshold for removing injections below newsnr_sg")
parser.add_argument("--injection-cluster-param",
                    help="Parameter to use when clustering injections")
parser.add_argument("--injection-cluster-window", type=float,
                    help="Window in seconds to use when clustering injections")

# Gather inputs for combining triggers and injection
group = parser.add_mutually_exclusive_group()
group.add_argument("--trigger-veto-window", type=float,
                   help="Window to remove injections around noise triggers")
group.add_argument("--injection-veto-window", type=float,
                   help="Window to remove noise triggers around injections")

# Gather additional options
parser.add_argument("--verbose", action='store_true')

args = parser.parse_args()

if args.verbose:
    log_level = logging.DEBUG
else:
    log_level = logging.WARNING
logging.basicConfig(format='%(asctime)s : %(message)s', level=log_level)

logging.info("Reading triggers from files.")
triggers = TriggerList(args.trigger_files)

triggers.apply_segments(args.segment_files, "TRIGGERS_GENERATED")
triggers.apply_segments(args.veto_files, "closed_box", within=False)

logging.info("Triggers available:")
for k, v in triggers.nums.items():
    logging.info("{0}: {1}".format(k, v))
    
args.trigger_newsnr_cut = max(args.trigger_newsnr_cut,
                              args.trigger_newsnr_sg_cut)
args.trigger_snr_cut = max(args.trigger_snr_cut,
                           args.trigger_newsnr_cut)

if args.trigger_snr_cut:
    triggers.threshold_cut(args.trigger_snr_cut, 'snr')
    
if args.trigger_newsnr_cut:
    triggers.get_newsnr()
    triggers.threshold_cut(args.trigger_newsnr_cut, 'newsnr')

if args.trigger_newsnr_sg_cut:
    triggers.get_newsnr_sg()
    triggers.threshold_cut(args.trigger_newsnr_sg_cut, 'newsnr_sg')

logging.info("Triggers available after threshold cuts:")
for k, v in triggers.nums.items():
    logging.info("{0}: {1}".format(k, v))

if args.trigger_cluster_param and args.trigger_cluster_window:
    triggers.cluster_over_time(args.trigger_cluster_window,
                               args.trigger_cluster_param)

logging.info("Triggers available after clustering:")
for k, v in triggers.nums.items():
    logging.info("{0}: {1}".format(k, v))

triggers.get_bank_params(args.bank_file)

logging.info("Reading injections from files.")
injections = InjectionTriggers(args.inj_find_files, args.inj_trigger_files,
                               args.inj_approximants, f_lowers=args.inj_f_lowers)

injections.apply_segments(args.segment_files, "TRIGGERS_GENERATED")
injections.apply_segments(args.veto_files, "closed_box", within=False)

logging.info("Injectionss available:")
for k, v in injections.nums.items():
    logging.info("{0}: {1}".format(k, v))

args.injection_newsnr_cut = max(args.injection_newsnr_cut,
                                args.injection_newsnr_sg_cut)
args.injection_snr_cut = max(args.injection_snr_cut,
                             args.injection_newsnr_cut)

if args.injection_snr_cut:
    injections.threshold_cut(args.injection_snr_cut, 'snr')
    
if args.injection_newsnr_cut:
    injections.get_newsnr()
    injections.threshold_cut(args.injection_newsnr_cut, 'newsnr')

if args.injection_newsnr_sg_cut:
    injections.get_newsnr_sg()
    injections.threshold_cut(args.injection_newsnr_sg_cut, 'newsnr_sg')

logging.info("Injections available after threshold cuts:")
for k, v in injections.nums.items():
    logging.info("{0}: {1}".format(k, v))

if args.injection_cluster_param and args.injection_cluster_window:
    injections.cluster_over_time(args.injection_cluster_window,
                               args.injection_cluster_param)

logging.info("Injections available after clustering:")
for k, v in injections.nums.items():
    logging.info("{0}: {1}".format(k, v))

injections.get_bank_params(args.bank_file)

if args.trigger_veto_window:
    times = {ifo: triggers.triggers[ifo]['end_time'] for ifo in triggers.ifos}
    injections.veto_times(times, args.trigger_veto_window)
    logging.info("Injections available after trigger veto:")
    for k, v in injections.nums.items():
        logging.info("{0}: {1}".format(k, v))
elif args.injection_veto_window:
    times = {ifo: injections.triggers[ifo]['end_time'] for ifo in injections.ifos}
    triggers.veto_times(times, args.injection_veto_window)
    logging.info("Triggers available after injection veto:")
    for k, v in triggers.nums.items():
        logging.info("{0}: {1}".format(k, v))

logging.info("Combining triggers and injections and saving")
triggers.append(injections)
triggers.write_to_hdf(args.output_sample_file)

logging.info("Creating injection file")
injections.write_injection_set(args.output_injection_file)

logging.info("Done!")
