import logging
import h5py
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


class SampleFile(object):

    def __init__(self, fp):
        self.fp = fp
        with h5py.File(fp, 'r') as f:
            self.delta_f = f.attrs['delta_f']
            self.flow = f.attrs['flow']
            self.flen = f.attrs['flen']
            self.tlen = f.attrs['tlen']
            self.sample_rate = f.attrs['sample_rate']
            self.num = f['stilde'].len()

    def get_samples(self, idxs):
        with h5py.File(self.fp, 'r') as f:
            segs = f['stilde'][idxs, :]
            psds = f['psd'][idxs, :]
            cuts = f['cut'][idxs, :]
            tids = f['template_id'][idxs]
            injs = f['injection'][idxs]
        return segs, psds, cuts, tids, injs

    def get_params(self, param, idxs):
        with h5py.File(self.fp, 'r') as f:
            if param == 'mtotal':
                return f['mass1'][idxs] + f['mass2'][idxs]
            else:
                return f[param][idxs]

    def get_param_min_max(self, param):
        idxs = np.arange(self.num)
        params = self.get_params(param, idxs)
        return np.min(params), np.max(params)

    def get_tensors(self, idxs, bank, shift_param):
        segs, psds, cuts, tids, injs = self.get_samples(idxs)
        temp = np.stack([bank[tid].numpy().copy() for tid in tids])
        gather_idxs = np.zeros((len(idxs), cuts[0, 1] - cuts[0, 0], 2), dtype=int)
        for k, cut in enumerate(cuts):
            gather_idxs[k, :, 0] = k
            gather_idxs[k, :, 1] = np.arange(cut[0], cut[1])
        param = self.get_params(shift_param, idxs)

        segs = tf.convert_to_tensor(segs, dtype=tf.complex64)
        psds = tf.convert_to_tensor(psds, dtype=tf.float32)
        temp = tf.convert_to_tensor(temp, dtype=tf.complex64)
        injs = tf.convert_to_tensor(injs, dtype=tf.bool)
        gather_idxs = tf.convert_to_tensor(gather_idxs, dtype=tf.int64)
    
        param = tf.convert_to_tensor(param, dtype=tf.float32)
        
        return segs, psds, temp, injs, gather_idxs, param


def chi2(x, df):
    term1 = tf.math.log(x) * (0.5 * df - 1)
    term2 = -0.5 * x
    term3 = tf.math.lgamma(0.5 * df) + tf.math.log(2.) * (0.5 * df)
    return term1 + term2 - term3


def nc_chi2(x, df, nc):
    # Implementation copied from scipy.stats.ncx2
    df2 = df / 2. - 1.
    xs = tf.math.sqrt(x)
    ns = tf.math.sqrt(nc)
    res = tf.math.xlogy(df2 / 2., x / nc) - 0.5 * (xs - ns) ** 2.
    corr = tfp.math.log_bessel_ive(df2, xs * ns) - tf.math.log(2.)
    return res + corr

