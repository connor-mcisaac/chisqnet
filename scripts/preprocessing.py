import h5py
import numpy as np
from pycbc.events.ranking import newsnr_sgveto, newsnr
from pycbc.events.coinc import cluster_over_time
from pycbc.events import veto
from pycbc.detector import Detector
from ligo.segments import segment, segmentlist
from pycbc.frame import frame_paths, read_frame
from pycbc.filter import resample_to_delta_t
from pycbc.psd import interpolate, inverse_spectrum_truncation
from pycbc.waveform import get_fd_waveform, apply_fseries_time_shift


_trigger_params = ['snr', 'chisq', 'chisq_dof', 'sg_chisq', 'end_time', 'template_id']
_trigger_dtypes = ['f8', 'f8', 'i4', 'f8', 'f8', 'i4']

_bank_params = ['mass1', 'mass2', 'spin1z', 'spin2z', 'template_duration']
_bank_dtypes = ['f8', 'f8', 'f8', 'f8', 'f8']

_temp_params = ['mass1', 'mass2', 'spin1z', 'spin2z', 'approximant']

_inj_in_params = ['mass1', 'mass2',
                  'spin1x', 'spin1y', 'spin1z',
                  'spin2x', 'spin2y', 'spin2z',
                  'distance', 'coa_phase', 'inclination']
_inj_in_dtypes = ['f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8']

_inj_ex_params = ['end_time', 'latitude', 'longitude', 'polarization']
_inj_ex_dtypes = ['f8', 'f8', 'f8', 'f8']

_sim_to_hdf = {'latitude': 'dec', 'longitude': 'ra', 'end_time': 'tc'}


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
        
        self._params = _trigger_params[:] + ['injection']
        self._dtypes = _trigger_dtypes[:] + ['?']
        self.triggers = {}
        self.nums = {}
        
        for trigger_file in trigger_files:

            with h5py.File(trigger_file, 'r') as f:

                ifo = list(f.keys())[0]
                if ifo not in self.triggers.keys():
                    self.triggers[ifo] = []
                    self.nums[ifo] = 0

                num = len(f[ifo][self._params[0]][:])
                record = np.zeros(num, dtype={'names': self._params, 'formats': self._dtypes})

                for p in _trigger_params:
                    record[p] = f[ifo][p][:]
                record['injection'] = np.array([False] * num, dtype=bool)

                self.triggers[ifo].append(record)
                self.nums[ifo] += num

        self.ifos = list(self.triggers.keys())

        for ifo in self.ifos:
            self.triggers[ifo] = np.concatenate(self.triggers[ifo])

    def get_newsnr(self):
        for ifo in self.ifos:
            updates = []
            names = []
            dtypes = []
            if 'rchisq' not in self.triggers[ifo].dtype.names:
                rchisq = (self.triggers[ifo]['chisq']
                          / (self.triggers[ifo]['chisq_dof'] * 2. - 2))
                updates.append(rchisq)
                names.append('rchisq')
                dtypes.append('f8')
            else:
                rchisq = self.triggers[ifo]['rchisq']
                
            if 'newsnr' not in self.triggers[ifo].dtype.names:
                nsnr = newsnr(self.triggers[ifo]['snr'], rchisq)
                updates.append(nsnr)
                names.append('newsnr')
                dtypes.append('f8')

            if len(updates) > 0:
                self.triggers[ifo] = add_params(self.triggers[ifo], updates,
                                                names, dtypes)

    def get_newsnr_sg(self):
        for ifo in self.ifos:
            updates = []
            names = []
            dtypes = []
            if 'rchisq' not in self.triggers[ifo].dtype.names:
                rchisq = (self.triggers[ifo]['chisq']
                          / (self.triggers[ifo]['chisq_dof'] * 2. - 2))
                updates.append(rchisq)
                names.append('rchisq')
                dtypes.append('f8')
            else:
                rchisq = self.triggers[ifo]['rchisq']
                
            if 'newsnr_sg' not in self.triggers[ifo].dtype.names:
                newsnr_sg = newsnr_sgveto(self.triggers[ifo]['snr'], rchisq,
                                          self.triggers[ifo]['sg_chisq'])
                updates.append(newsnr_sg)
                names.append('newsnr_sg')
                dtypes.append('f8')

            if len(updates) > 0:
                self.triggers[ifo] = add_params(self.triggers[ifo], updates,
                                                names, dtypes)

    def threshold_cut(self, thresh, param):
        
        for ifo in self.ifos:
            lgc = self.triggers[ifo][param] > thresh
            self.triggers[ifo] = self.triggers[ifo][lgc]
            self.nums[ifo] = np.sum(lgc)

    def cluster_over_time(self, window, param):

        for ifo in self.ifos:
            idxs = cluster_over_time(self.triggers[ifo][param],
                                     self.triggers[ifo]['end_time'],
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
            times = self.triggers[ifo]['end_time']
            if within:
                idx, segs = veto.indices_within_segments(times, seg_files,
                                                         segment_name=seg_name, ifo=ifo)
            else:
                idx, segs = veto.indices_outside_segments(times, seg_files,
                                                          segment_name=seg_name, ifo=ifo)
            self.triggers[ifo] = self.triggers[ifo][idx]
            self.nums[ifo] = len(idx)

    def veto_times(self, times, window):
        for ifo in self.ifos:
            if isinstance(times, dict):
                time = times[ifo]
            else:
                time = times
            starts = time - window
            ends = time + window
            
            idxs = veto.indices_outside_times(self.triggers[ifo]['end_time'], starts, ends)
            self.triggers[ifo] = self.triggers[ifo][idxs]
            self.nums[ifo] = len(idxs)

    def get_time_cut(self, ifo, start, end):
        lgc = self.triggers[ifo]['end_time'] >= start
        lgc *= self.triggers[ifo]['end_time'] <= end
        trigs = self.triggers[ifo][lgc]
        return trigs

    def write_to_hdf(self, fp):
        with h5py.File(fp, 'w') as f:
            for ifo in self.ifos:
                i = f.create_group(ifo)
                for p, d in zip(self._params, self._dtypes):
                    data = self.triggers[ifo][p][:]
                    if d.startswith('U'):
                        data = data.astype('S')
                    _ = i.create_dataset(p, data=data)
            f.attrs['params'] = self._params
            f.attrs['dtypes'] = self._dtypes

    @classmethod
    def read_from_hdf(cls, fp):
        ob = cls.__new__(cls)
        ob.triggers = {}
        ob.nums = {}
        with h5py.File(fp, 'r') as f:
            ob._params = f.attrs['params']
            ob._dtypes = f.attrs['dtypes']
            ob.ifos = list(f.keys())
            for ifo in f.keys():
                num = None
                for p, d in zip(ob._params, ob._dtypes):
                    data = f[ifo][p][:]
                    if num is None:
                        num = len(data)
                        record = np.zeros(num, dtype={'names': ob._params,
                                                      'formats': ob._dtypes})
                    if d.startswith('U'):
                        data = data.astype('U')
                    record[p] = data
                ob.triggers[ifo] = record
                ob.nums[ifo] = num
        return ob

    def append(self, triggers):
        for ifo in self.ifos:
            if ifo not in triggers.triggers.keys():
                continue
            common = [p for p in self._params if p in triggers._params]
            self.triggers[ifo] = np.concatenate([self.triggers[ifo][common],
                                                 triggers.triggers[ifo][common]])
            self.nums[ifo] = len(self.triggers[ifo])


class InjectionTriggers(TriggerList):
    
    def __init__(self, inj_finds, inj_trigs, approximants, f_lowers=None):

        self._inj_params = _inj_in_params[:] + _inj_ex_params[:]
        self._inj_dtypes = _inj_in_dtypes[:] + _inj_ex_dtypes[:]

        self._trig_params = _trigger_params[:]
        self._trig_dtypes = _trigger_dtypes[:]

        self._params = ['inj:' + p for p in self._inj_params + ['approximant', 'f_lower']]
        self._params += self._trig_params + ['injection', 'injection_index']
        self._dtypes = self._inj_dtypes + ['U16', 'f8'] + self._trig_dtypes + ['?', 'i4']

        self.triggers = {}
        self.nums = {}

        if f_lowers is None:
            f_lowers = [10.] * len(inj_finds)

        for inj_find, inj_trig, approx, f_lower in zip(inj_finds, inj_trigs, approximants, f_lowers):

            inj_trigs = inj_trig.split(',')
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
                        record['injection'] = np.array([True] * num, dtype=bool)

                    record['inj:approximant'] = np.array([approx] * num, dtype='U16')
                    record['inj:f_lower'] = np.array([f_lower] * num, dtype='f8')
                    record['injection_index'] = inj_id

                    records[ifo] = record

            for ifo in ifos:
                with h5py.File(inj_trigs[ifo], 'r') as f:
                    for p in self._trig_params:
                        records[ifo][p] = f[ifo][p][:][trig_ids[ifo]]

                self.triggers[ifo].append(records[ifo])
        
        self.ifos = list(self.triggers.keys())

        for ifo in self.ifos:
            self.triggers[ifo] = np.concatenate(self.triggers[ifo])

    def write_to_hdf(self, fp):
        super().write_to_hdf(fp)
        with h5py.File(fp, 'a') as f:
            f.attrs['inj_params'] = self._inj_params
            f.attrs['inj_dtypes'] = self._inj_dtypes
            f.attrs['trig_params'] = self._trig_params
            f.attrs['trig_dtypes'] = self._trig_dtypes

    @classmethod
    def read_from_hdf(cls, fp):
        ob = super().read_from_hdf(fp)
        with h5py.File(fp, 'r') as f:
            ob._inj_params = f.attrs['inj_params']
            ob._inj_dtypes = f.attrs['inj_dtypes']
            ob._trig_params = f.attrs['trig_params']
            ob._trig_dtypes = f.attrs['trig_dtypes']
        return ob

    def write_injection_set(self, fp, **kwargs):

        with h5py.File(fp, 'w') as f:
            f.attrs['injtype'] = 'cbc'
            for ifo in self.ifos:
                g = f.create_group(ifo)
                g.attrs["static_args"] = list(kwargs.keys())
                for k, v in kwargs.items():
                    if isinstance(v, bytes):
                        g.attrs[k] = str(v)
                    else:
                        g.attrs[k] = v
                for p in self._inj_params + ['approximant', 'f_lower']:
                    if p in _sim_to_hdf.keys():
                        n = _sim_to_hdf[p]
                    else:
                        n = p
                    if self.triggers[ifo]['inj:' + p].dtype.char == 'U':
                        g[n] = self.triggers[ifo]['inj:' + p].astype('S')
                    else:
                        g[n] = self.triggers[ifo]['inj:' + p]


def gather_segments(seg_files, seg_name, ifo):
    segs = segmentlist([])
    for f in seg_files:
        segs += veto.select_segments_by_definer(f, seg_name, ifo)
    segs.coalesce()
    return segs
