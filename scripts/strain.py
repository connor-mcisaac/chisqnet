import sys
import argparse
from glue import lal
from glue.ligolw import table, lsctables, ligolw
from glue.ligolw import utils as ligolw_utils
from glue.ligolw.utils import segments as ligolw_segments
from glue.ligolw.utils import process as ligolw_process
from ligo.segments import segment, segmentlist, segmentlistdict, infinity
from pycbc.inject import InjectionSet
from pycbc.types import MultiDetOptionAction
import copy
import logging, numpy
import pycbc.types
from pycbc.strain import detect_loud_glitches, gate_data, StrainSegments
from pycbc.filter import resample_to_delta_t, highpass
import pycbc.frame


# segments functions adapted from SegFile at:
# https://github.com/gwastro/pycbc/blob/master/pycbc/workflow/core.py
class ContentHandler(ligolw.LIGOLWContentHandler):
    pass


lsctables.use_in(ContentHandler)


def segmentlistdict_to_xml(segments, output):
    outdoc = ligolw.Document()
    outdoc.appendChild(ligolw.LIGO_LW())
    process = ligolw_process.register_to_xmldoc(outdoc, sys.argv[0], {})

    valid_segments = segments.extent_all()
    for key, seglist in segments.items():
        ifo, name = key.split(':')

        fsegs = [(lal.LIGOTimeGPS(seg[0]),
                  lal.LIGOTimeGPS(seg[1])) for seg in seglist]

        vsegs = [(lal.LIGOTimeGPS(valid_segments[0]),
                  lal.LIGOTimeGPS(valid_segments[1]))]

        with ligolw_segments.LigolwSegments(outdoc, process) as x:
            x.add(ligolw_segments.LigolwSegmentList(active=fsegs,
                                                    instruments=set([ifo]), name=name,
                                                    version=1, valid=vsegs))

        ligolw_utils.write_filename(outdoc, output)


def xml_to_segmentlistdict(xml):
    # load xmldocument and SegmentDefTable and SegmentTables
    fp = open(xml, 'rb')
    xmldoc, _ = ligolw_utils.load_fileobj(fp,
                                          gz=xml.endswith(".gz"),
                                          contenthandler=ContentHandler)

    seg_def_table = table.get_table(xmldoc,
                                    lsctables.SegmentDefTable.tableName)
    seg_table = table.get_table(xmldoc, lsctables.SegmentTable.tableName)

    segs = segmentlistdict()

    seg_id = {}
    for seg_def in seg_def_table:
        # Here we want to encode ifo and segment name
        full_channel_name = ':'.join([str(seg_def.ifos),
                                      str(seg_def.name)])
        seg_id[int(seg_def.segment_def_id)] = full_channel_name
        segs[full_channel_name] = segmentlist()
        
    for seg in seg_table:
        seg_obj = segment(
            lal.LIGOTimeGPS(seg.start_time, seg.start_time_ns),
            lal.LIGOTimeGPS(seg.end_time, seg.end_time_ns))
        segs[seg_id[int(seg.segment_def_id)]].append(seg_obj)

    return segs


class InjectionSetMulti(InjectionSet):

    @staticmethod
    def from_cli(opt, ifo):
        """Return an instance of InjectionSet configured as specified
        on the command line.
        """
        if opt.injection_file is None:
            return None

        kwa = {'hdf_group': ifo}
        return InjectionSet(opt.injection_file, **kwa)


class TimeMapping(object):
    
    def __init__(self, string):
        self.segments = segmentlist([])
        self.options = []
        strings = string.split(',')
        for op in strings:
            if '[' not in op:
                self.options.append(op)
                seg = segment(0, infinity())
            else:
                self.options.append(op.split('[')[0])
                times = op.split('[')[1].split(']')[0]
                start, end = times.split(':')
                seg = segment(int(start), int(end))
            if self.segments.intersects_segment(seg):
                raise ValueError("There should be no overlapping segments "
                                 + "for time varying arguments")
            self.segments.append(seg)

    def __call__(self, time=1):
        idx = self.segments.find(time)
        return self.options[idx]


class MultiDetTimeOptionAction(MultiDetOptionAction):
    
    def __call__(self, parser, namespace, values, option_string=None):
        err_msg = "Issue with option: %s \n" %(self.dest,)
        err_msg += "Received value: %s \n" %(' '.join(values),)
        if getattr(namespace, self.dest, None) is None:
            setattr(namespace, self.dest, {})
        items = getattr(namespace, self.dest)
        items = copy.copy(items)
        for value in values:
            det = value.split(':')[0]
            try:
                val = TimeMapping(value[len(det) + 1:])
            except Exception as e:
                err_msg += e.message
                raise ValueError(err_msg)
            items[det] = val
        setattr(namespace, self.dest, items)


class StrainSegmentsCut(StrainSegments):

    def cut(self, times, cut_width):

        segment_slices_red = []
        analyze_slices_red = []
        for seg, ana in zip(self.segment_slices, self.analyze_slices):
            start_time = int(self.strain.start_time) + (seg.start + ana.start) * self.delta_t
            end_time = int(self.strain.start_time) + (seg.start + ana.stop) * self.delta_t
            lgc = times >= start_time
            lgc *= times < end_time
            if numpy.any(lgc):
                segment_slices_red.append(seg)
                analyze_slices_red.append(ana)
        self.segment_slices = segment_slices_red
        self.analyze_slices = analyze_slices_red

        cut_width = int(cut_width * self.sample_rate)
        sample_segments = []
        sample_slices = []
        for time in times:
            tidx = int((time - int(self.strain.start_time)) * self.sample_rate)
            start = tidx - cut_width
            end = tidx + cut_width + 1
            for i in range(len(self.segment_slices)):
                seg = self.segment_slices[i]
                ana = self.analyze_slices[i]
                ana_start = seg.start + ana.start
                ana_end = seg.start + ana.stop
                if tidx < ana_start or tidx >= ana_end:
                    continue
                if start < ana_start:
                    start = ana_start
                    end = start + 2 * cut_width + 1
                elif end > ana_end:
                    end = ana_end
                    start = end - 2 * cut_width - 1
                sample_segments.append(i)
                sample_slices.append(slice(start - seg.start, end - seg.start))
                break
        self.sample_segments = sample_segments
        self.sample_slices = sample_slices


def strain_from_cli(ifo, segment, opt,
                    dyn_range_fac=1, precision='single'):
    gating_info = {}

    injector = InjectionSetMulti.from_cli(opt, ifo)
    
    frame_type = opt.frame_type[ifo](segment)
    channel_name = opt.channel_name[ifo](segment)

    gps_start_time = segment[0] - opt.segment_start_pad
    gps_end_time = segment[1] + opt.segment_end_pad

    logging.info("Reading Frames")

    if hasattr(opt, 'frame_sieve') and opt.frame_sieve:
        sieve = opt.frame_sieve
    else:
        sieve = None

    strain = pycbc.frame.query_and_read_frame(
        frame_type, channel_name,
        start_time=gps_start_time-opt.pad_data,
        end_time=gps_end_time+opt.pad_data,
        sieve=sieve)

    if opt.normalize_strain:
        logging.info("Dividing strain by constant")
        l = opt.normalize_strain
        strain = strain / l

    if opt.strain_high_pass:
        logging.info("Highpass Filtering")
        strain = highpass(strain, frequency=opt.strain_high_pass)

    if opt.sample_rate:
        logging.info("Resampling data")
        strain = resample_to_delta_t(strain,
                                     1. / opt.sample_rate,
                                     method='ldas')

    if injector is not None:
        logging.info("Applying injections")
        injections = \
            injector.apply(strain, ifo,
                           distance_scale=opt.injection_scale_factor,
                           injection_sample_rate=opt.injection_sample_rate)

    if precision == 'single':
        logging.info("Converting to float32")
        strain = (strain * dyn_range_fac).astype(pycbc.types.float32)
    elif precision == "double":
        logging.info("Converting to float64")
        strain = (strain * dyn_range_fac).astype(pycbc.types.float64)
    else:
        raise ValueError("Unrecognized precision {}".format(precision))

    if opt.gating_file is not None:
        logging.info("Gating times contained in gating file")
        gate_params = numpy.loadtxt(opt.gating_file)
        if len(gate_params.shape) == 1:
            gate_params = [gate_params]
        strain = gate_data(strain, gate_params)
        gating_info['file'] = \
                [gp for gp in gate_params \
                 if (gp[0] + gp[1] + gp[2] >= strain.start_time) \
                 and (gp[0] - gp[1] - gp[2] <= strain.end_time)]

    if opt.autogating_threshold is not None:
        gating_info['auto'] = []
        for _ in range(opt.autogating_max_iterations):
            glitch_times = detect_loud_glitches(
                    strain, threshold=opt.autogating_threshold,
                    cluster_window=opt.autogating_cluster,
                    low_freq_cutoff=opt.strain_high_pass,
                    corrupt_time=opt.pad_data + opt.autogating_pad)
            gate_params = [[gt, opt.autogating_width, opt.autogating_taper]
                           for gt in glitch_times]
            gating_info['auto'] += gate_params
            strain = gate_data(strain, gate_params)
            if len(glitch_times) > 0:
                logging.info('Autogating at %s',
                             ', '.join(['%.3f' % gt
                                        for gt in glitch_times]))
            else:
                break

    if opt.strain_high_pass:
        logging.info("Highpass Filtering")
        strain = highpass(strain, frequency=opt.strain_high_pass)

    if opt.pad_data:
        logging.info("Remove Padding")
        start = int(opt.pad_data * strain.sample_rate)
        end = int(len(strain) - strain.sample_rate * opt.pad_data)
        strain = strain[start:end]

    if opt.taper_data:
        logging.info("Tapering data")
        # Use auto-gating, a one-sided gate is a taper
        pd_taper_window = opt.taper_data
        gate_params = [(strain.start_time, 0., pd_taper_window)]
        gate_params.append((strain.end_time, 0., pd_taper_window))
        gate_data(strain, gate_params)

    if injector is not None:
        strain.injections = injections
    strain.gating_info = gating_info

    return strain


def insert_strain_option_group(parser):
    """ Add strain-related options to the optparser object.
    Adds the options used to call the pycbc.strain.from_cli function to an
    optparser as an OptionGroup. This should be used if you
    want to use these options in your code.
    Parameters
    -----------
    parser : object
        OptionParser instance.
    """

    data_reading_group = parser.add_argument_group("Options for obtaining h(t)",
                  "These options are used for generating h(t) either by "
                  "reading from a file or by generating it. This is only "
                  "needed if the PSD is to be estimated from the data, ie. "
                  " if the --psd-estimation option is given.")

    # Required options
    data_reading_group.add_argument("--strain-high-pass", type=float, required=True,
                                    help="High pass frequency")
    data_reading_group.add_argument("--pad-data", default=8,
                                    help="Extra padding to remove highpass corruption "
                                    "(integer seconds)", type=int)
    data_reading_group.add_argument("--taper-data",
                                    help="Taper ends of data to zero using the supplied length as a "
                                    "window (integer seconds)", type=int, default=0)
    data_reading_group.add_argument("--sample-rate", type=int, required=True,
                                    help="The sample rate to use for h(t) generation (integer Hz).")
    data_reading_group.add_argument("--channel-name", type=str, required=True,
                                    help="The channel containing the gravitational strain data",
                                    nargs = '+', action=MultiDetTimeOptionAction)
    data_reading_group.add_argument("--frame-type", type=str, required=True,
                                    help="(optional), replaces frame-files. Use datafind "
                                    "to get the needed frame file(s) of this type.",
                                    nargs = '+', action=MultiDetTimeOptionAction)

    # Filter frame files by URL
    data_reading_group.add_argument("--frame-sieve",
                            type=str,
                            help="(optional), Only use frame files where the "
                                 "URL matches the regular expression given.")

    # Injection options
    data_reading_group.add_argument("--injection-file", type=str,
                      help="(optional) Injection file used to add "
                           "waveforms into the strain")
    data_reading_group.add_argument("--injection-scale-factor", type=float,
                    default=1, help="Divide injections by this factor "
                    "before injecting into the data.")
    data_reading_group.add_argument("--injection-sample-rate", type=float,
                    help="Sample rate for injections")

    # Gating options
    data_reading_group.add_argument("--gating-file", type=str,
                    help="(optional) Text file of gating segments to apply."
                        " Format of each line is (all times in secs):"
                        "  gps_time zeros_half_width pad_half_width")
    data_reading_group.add_argument('--autogating-threshold', type=float,
                                    metavar='SIGMA',
                                    help='If given, find and gate glitches '
                                         'producing a deviation larger than '
                                         'SIGMA in the whitened strain time '
                                         'series.')
    data_reading_group.add_argument('--autogating-max-iterations', type=int,
                                    metavar='SIGMA', default=1,
                                    help='If given, iteratively apply '
                                         'autogating')
    data_reading_group.add_argument('--autogating-cluster', type=float,
                                    metavar='SECONDS', default=5.,
                                    help='Length of clustering window for '
                                         'detecting glitches for autogating.')
    data_reading_group.add_argument('--autogating-width', type=float,
                                    metavar='SECONDS', default=0.25,
                                    help='Half-width of the gating window.')
    data_reading_group.add_argument('--autogating-taper', type=float,
                                    metavar='SECONDS', default=0.25,
                                    help='Taper the strain before and after '
                                         'each gating window over a duration '
                                         'of SECONDS.')
    data_reading_group.add_argument('--autogating-pad', type=float,
                                    metavar='SECONDS', default=16,
                                    help='Ignore the given length of whitened '
                                         'strain at the ends of a segment, to '
                                         'avoid filters ringing.')
    # Optional
    data_reading_group.add_argument("--normalize-strain", type=float,
                    help="(optional) Divide frame data by constant.")

    return data_reading_group

