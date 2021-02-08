import h5py
import numpy as np
from pycbc.events.ranking import newsnr_sgveto
from pycbc.events.eventmgr import findchirp_cluster_over_window
from pycbc.events import veto
from pycbc.detector import Detector
from ligo.segments import segment, segmentlist
from pycbc.frame import frame_paths, read_frame
from pycbc.filter import resample_to_delta_t
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform import get_fd_waveform, apply_fseries_time_shift
import glob


_trigger_params = ['snr', 'chisq', 'chisq_dof', 'sg_chisq', 'end_time', 'template_id']
_trigger_dtypes = ['f8', 'f8', 'i4', 'f8', 'f8', 'i4']

_bank_params = ['mass1', 'mass2', 'spin1z', 'spin2z']
_bank_dtypes = ['f8', 'f8', 'f8', 'f8', 'f8']

_inj_in_params = ['mass1', 'mass2',
                  'spin1x', 'spin1y', 'spin1z',
                  'spin2x', 'spin2y', 'spin2z',
                  'distance', 'coa_phase', 'inclination']
_inj_in_dtypes = ['f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8']

_inj_ex_params = ['end_time', 'latitude', 'longitude', 'polarization']
_inj_ex_dtypes = ['f8', 'f8', 'f8', 'f8']


def add_params(record, updates, names, dtypes):
    dt = [(n, d) for n, d in zip(names, dtypes)]
    updated = np.zeros(len(record), dtype=record.dtype.descr + dt)
    for p in record.dtype.names:
        updated[p] = record[p]
    for u, n in zip(updates, names):
        updated[n] = u
    return updated


class TriggerList(object):

    def __init__(self, trigger_files):
        
        self._params = _trigger_params[:]
        self._dtypes = _trigger_dtypes[:]
        self.triggers = {}
        self.nums = {}
        self.newsnr = False
        
        for trigger_file in trigger_files:

            with h5py.File(trigger_file, 'r') as f:

                ifo = list(f.keys())[0]
                if ifo not in self.triggers.keys():
                    self.triggers[ifo] = []
                    self.nums[ifo] = 0

                num = len(f[ifo][self._params[0]][:])
                record = np.zeros(num, dtype={'names': self._params, 'formats': self._dtypes})

                for p in self._params:
                    record[p] = f[ifo][p][:]

                self.triggers[ifo].append(record)
                self.nums[ifo] += num

        self.ifos = list(self.triggers.keys())

        for ifo in self.ifos:
            self.triggers[ifo] = np.concatenate(self.triggers[ifo])
        

    def get_newsnr(self):
        if not self.newsnr:
            for ifo in self.ifos:
                self.triggers[ifo]['chisq'] = (self.triggers[ifo]['chisq']
                                               / (self.triggers[ifo]['chisq_dof'] * 2. - 2))
        
                newsnr_sg = newsnr_sgveto(self.triggers[ifo]['snr'],
                                          self.triggers[ifo]['chisq'],
                                          self.triggers[ifo]['sg_chisq'])

                self.triggers[ifo] = add_params(self.triggers[ifo], [newsnr_sg],
                                                ['newsnr_sg'], ['f8'])
            self.newsnr = True

    def threshold_cut(self, thresh, param):
        
        for ifo in self.ifos:
            lgc = self.triggers[ifo][param] > thresh
            self.triggers[ifo] = self.triggers[ifo][lgc]
            self.nums[ifo] = np.sum(lgc)

    def cluster_over_time(self, window, param):

        for ifo in self.ifos:
            idxs = findchirp_cluster_over_window(self.triggers[ifo]['end_time'],
                                                 self.triggers[ifo][param],
                                                 window)
            self.triggers[ifo] = self.triggers[ifo][idxs]
            self.nums[ifo] = len(idxs)

    def get_bank_params(self, bank_file):

        self._params += _bank_params[:] + ['approximant']
        self._dtypes += _bank_dtypes[:] + ['U12']
        
        with h5py.File(bank_file, 'r') as f:
            for ifo in self.ifos:

                update = [f[p][:][self.triggers[ifo]['template_id']] for p in _bank_params]
                self.triggers[ifo] = add_params(self.triggers[ifo], update,
                                                _bank_params, _bank_dtypes)

                lgc = (self.triggers[ifo]['mass1'] + self.triggers[ifo]['mass2']) >= 4
                update = np.where(lgc, 'SEOBNRv4_ROM', 'TaylorF2')
                self.triggers[ifo] = add_params(self.triggers[ifo], [update],
                                                ['approximant'], ['U12'])

    def flatten(self):
        records = []
        ifos = []
        for ifo in self.ifos:
            records.append(self.triggers[ifo])
            ifos.append(np.array([ifo] * self.nums[ifo], dtype='U2'))
        record = np.concatenate(records)
        ifos = np.concatenate(ifos)

        flat = add_params(record, [ifos], ['ifo'], ['U2'])

        return flat

    def draw_ifos(self, n=1):
        num = np.array([self.nums[ifo] for ifo in self.ifos], dtype=np.float64)
        prob = num/np.sum(num)
        return np.random.choice(self.ifos, size=n, p=prob)

    def draw_triggers(self, n=1, ifo=None):
        if ifo is not None:
            draw = np.random.choice(self.triggers[ifo], size=n)
            ifos = np.array([ifo] * n, dtype='U2')
            draw = add_params(draw, [ifos], ['ifo'], ['U2'])
        else:
            flat = self.flatten()
            draw = np.random.choice(flat, size=n)
        return draw

    def apply_segments(self, seg_files, seg_name, within=True):
        for ifo in self.ifos:
            times = self.triggers[ifo]['end_time'][:]
            if within:
                idx, segs = veto.indices_within_segments(times, seg_files,
                                                         segment_name=seg_name, ifo=ifo)
            else:
                idx, segs = veto.indices_outside_segments(times, seg_files,
                                                          segment_name=seg_name, ifo=ifo)
            self.triggers[ifo] = self.triggers[ifo][idx]
            self.nums[ifo] = len(idx)

    def get_time_cut(self, ifo, start, end):
        lgc = self.triggers[ifo]['end_time'] >= start
        lgc *= self.triggers[ifo]['end_time'] <= end
        trigs = self.triggers[ifo][lgc]
        return trigs


class InjectionTriggers(TriggerList):
    
    def __init__(self, inj_dirs, approximants,
                 find_re="*-HDFINJFIND_*_INJ_INJECTIONS-*-*.hdf",
                 trig_re="?1-HDF_TRIGGER_MERGE_*_INJ_INJECTIONS-*-*.hdf"):

        self._inj_params = _inj_in_params[:] + _inj_ex_params[:]
        self._inj_dtypes = _inj_in_dtypes[:] + _inj_ex_dtypes[:]

        self._trig_params = _trigger_params[:]
        self._trig_dtypes = _trigger_dtypes[:]

        self._params = ['inj:' + p for p in self._inj_params + ['approximant']] + self._trig_params
        self._dtypes = self._inj_dtypes + ['U16'] + self._trig_dtypes

        self.triggers = {}
        self.nums = {}
        self.newsnr = False

        for inj_dir, approx in zip(inj_dirs, approximants):

            inj_find = glob.glob(inj_dir + '/' + find_re)[0]
            inj_trigs = glob.glob(inj_dir + '/' + trig_re)
            inj_trigs = {trig.split('/')[-1][:2]: trig for trig in inj_trigs}

            trig_ids = {}
            records = {}

            with h5py.File(inj_find, 'r') as f:
                inj_ids = f['/found_after_vetoes/injection_index'][:]
                ifos = f.attrs['ifos'].split(' ')

                for ifo in ifos:
                    trig_id = f['/found_after_vetoes/' + ifo + '/trigger_id'][:]
                    lgc = trig_id != -1
                    
                    trig_ids[ifo] = trig_id[lgc]
                    inj_id = inj_ids[lgc]

                    if ifo not in self.triggers.keys():
                        self.triggers[ifo] = []
                        self.nums[ifo] = 0

                    num = len(trig_ids[ifo])
                    self.nums[ifo] += num

                    record = np.zeros(num, dtype={'names': self._params, 'formats': self._dtypes})

                    for p in self._inj_params:
                        record['inj:' + p] = f['injections/' + p][inj_id]

                    record['inj:approximant'] = np.array([approx] * num, dtype='U16')

                    records[ifo] = record

            for ifo in ifos:
                with h5py.File(inj_trigs[ifo], 'r') as f:
                    for p in self._trig_params:
                        records[ifo][p] = f[ifo][p][:][trig_ids[ifo]]

                self.triggers[ifo].append(records[ifo])
        
        self.ifos = list(self.triggers.keys())

        for ifo in self.ifos:
            self.triggers[ifo] = np.concatenate(self.triggers[ifo])
        

def gather_segments(seg_files, seg_name, ifo):
    segs = segmentlist([])
    for f in seg_files:
        segs += veto.select_segments_by_definer(f, seg_name, ifo)
    segs.coalesce()
    return segs


class TemplateGen(object):
    
    def __init__(self, f_lower, delta_f, length):
        self.f_lower = f_lower
        self.delta_f = delta_f
        self.length = length

    def generate_template(self, trig):
        params = {p: trig[p] for p in _bank_params + ['approximant']}
        h, _ = get_fd_waveform(params,
                               delta_f=self.delta_f,
                               f_lower=self.f_lower)
        h.resize(self.length)
        return h

    def generate_injection(self, trig, ifo=None):
        params = {p: trig['inj:' + p] for p in _inj_in_params + ['approximant']}
        hp, hc = get_fd_waveform(params,
                                 delta_f=self.delta_f,
                                 f_lower=self.f_lower)

        time = trig['inj:end_time']
        ra = trig['inj:longitude']
        dec = trig['inj:latitude']
        pol = trig['inj:polarization']

        if ifo is None:
            det = Detector(trig['ifo'])
        else:
            det = Detector(ifo)

        fp, fc = det.antenna_pattern(ra, dec, pol, time)
        h = fp * hp + fc * hc
        
        dt = det.time_delay_from_earth_center(ra, dec, time)

        h._epoch = time + dt
        h.resize(self.length)
        return h

    def make_injection(self, strain, trig, ifo=None):
        injection = self.generate_injection(trig, ifo=ifo)
        injection = apply_fseries_time_shift(injection,
                                             injection._epoch - strain._epoch)
        strain = strain.astype(np.complex128)
        injection = injection.astype(np.complex128)
        return strain + injection

class StrainGen(object):
    
    def __init__(self, ifo, segment_length, frame, channel, segment_files,
                 sample_rate=2048, start_pad=16., end_pad=16., psd_width=16.):
        self.segment_length = segment_length
        self.frame = frame
        self.channel = channel
        self.sample_rate = sample_rate
        self.start_pad = start_pad
        self.end_pad = end_pad
        self.psd_width = psd_width
        self.time = None
        self.times = None
        self.padded = None
        self.strain = None
        self.stilde = None
        self.psd = None
        self.segments = gather_segments(segment_files, 'DATA_ANALYSED', ifo)

    def set_times(self, time):
        left = time - self.start_pad
        while True:
            right = left + self.segment_length + self.start_pad + self.end_pad
            if segment(left, right) in self.segments:
                break
            elif left + self.segment_length < time:
                raise ValueError("GPS time {0} cannot be fit within a valid segment".format(time))
            left -= 0.01
        self.padded = (left, right)
        self.times = (left + self.start_pad, right - self.end_pad)

    def get_strain(self, time):
        self.set_times(time)
        path = frame_paths(self.frame,
                           int(np.floor(self.padded[0])),
                           int(np.ceil(self.padded[1])))
        strain = read_frame(path, self.channel,
                            start_time=self.padded[0],
                            end_time=self.padded[1],
                            sieve=None, check_integrity=False)
        strain = resample_to_delta_t(strain, 1.0/self.sample_rate)
        self.strain = strain
        self.stilde = None
        self.psd = None
        return strain

    def get_stilde(self):
        stilde = self.strain.to_frequencyseries()
        self.stilde = stilde
        return stilde

    def get_psd(self):
        psd = self.strain.psd(self.psd_width)
        psd = interpolate(psd, self.strain.delta_f)
        psd = inverse_spectrum_truncation(psd, int(self.strain.sample_rate * self.psd_width),
                                          low_frequency_cutoff=15)
        self.psd = psd
        return psd

    def get_cut(self, time, width):
        if time < self.times[0] or time > self.times[1]:
            raise ValueError("time must be within the loaded segment")
        left = min(time, self.times[1] - width)
        right = left + width
        strain = self.strain.time_slice(left - self.start_pad,
                                        right + self.end_pad)
        stilde = strain.to_frequencyseries()
        return strain, stilde

    def get_interp_psd(self, delta_f):
        psd = interpolate(self.psd, delta_f)
        return psd


class BatchGen(object):
    
    def __init__(self, ifos, triggers, injections, templates, batchsize,
                 cut_width=16., snr_width=0.1):
        self.ifos = ifos
        self.triggers = triggers
        self.injections = injections
        self.templates = templates
        self.batchsize = batchsize
        self.cut_width = cut_width
        self.snr_width = snr_width

    def get_set(self, max_size):
        base = self.triggers.draw_triggers()
        ifo = base['ifo'][0]
        time = base['end_time'][0]
        
        strain = self.ifos[ifo].get_strain(time)
        psd = self.ifos[ifo].get_psd().numpy()

        start = self.ifos[ifo].times[0]
        end = self.ifos[ifo].times[1]

        all_trigs = self.triggers.get_time_cut(ifo, start, end)
        all_injs = self.injections.get_time_cut(ifo, start, end)

        all_triggers = [t for t in all_trigs] + [i for i in all_injs]
        np.random.shuffle(all_triggers)

        num = min(len(all_triggers), max_size)
        selected = all_triggers[:num]

        stildes = []
        psds = []
        templates = []
        params = []
        labels = []
        cut_idxs = []

        for trig in selected:
            strain, stilde = self.ifos[ifo].get_cut(trig['end_time'], self.cut_width)
            if 'inj:approximant' in trig.dtype.names:
                stilde = self.templates.make_injection(stilde, trig, ifo=ifo)
                labels.append(True)
            else:
                labels.append(False)
            stildes.append(stilde)

            psd = self.ifos[ifo].get_interp_psd(stilde.delta_f)
            psds.append(psd)

            template = self.templates.generate_template(trig)
            templates.append(template)

            params.append(trig[_trigger_params + _bank_params])
            
            times = strain.sample_times
            left = np.argmax(times > (trig['end_time'] - self.snr_width / 2.))
            right = left + int(self.snr_width * self.ifos[ifo].sample_rate)
            cut_idxs.append(np.array([left, right]))

        return stildes, psds, templates, params, labels, cut_idxs

    def get_batch(self):
        stildes = []
        psds = []
        templates = []
        params = []
        labels = []
        cut_idxs = []

        while len(stildes) < self.batchsize:
            remaining = self.batchsize - len(stildes)
            stilde, psd, temp, param, label, cut = self.get_set(remaining)
            stildes += stilde
            psds += psd
            templates += temp
            params += param
            labels += label
            cut_idxs += cut

        stildes = np.stack(stildes)
        psds = np.stack(psds)
        templates = np.stack(templates)
        params = np.stack(params)
        labels = np.array(labels)
        cut_idxs = np.stack(cut_idxs)

        return stildes, psds, templates, params, labels, cut_idxs
