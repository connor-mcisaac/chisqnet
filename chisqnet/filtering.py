import logging
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def real_to_complex(real):
    imag = tf.zeros_like(real, dtype=tf.float32)
    return tf.complex(real, imag)


def add_temp_axis(func):
    def _add_temp_axis(obj, *args, **kwargs):
        new_args = []
        for arg in args:
            if tf.rank(arg) == 2:
                arg = tf.expand_dims(arg, axis=1)
            new_args.append(arg)
        return func(obj, *new_args, **kwargs)
    return _add_temp_axis


class BaseFilter(object):

    def __init__(self, freqs, f_low, f_high):

        self.freqs = tf.convert_to_tensor(freqs, tf.float32)
        self.f_low = f_low
        self.f_high = f_high

        self.kmin = np.argmax(freqs >= f_low)
        if f_high < freqs[-1]:
            self.kmax = np.argmax(freqs > f_high)
        else:
            self.kmax = len(freqs)

        self.delta_f = freqs[1] - freqs[0]
        self.delta_t = 1. / freqs[-1] / 2.

        self.length = len(freqs)
        self.full_length = 2 * (self.length - 1)
        self.cut_length = self.kmax - self.kmin

        self.times = tf.range(self.full_length, delta=1., dtype=tf.float32) * self.delta_t

    def cut(self, x):
        return x[..., self.kmin:self.kmax]

    def pad(self, x, full=False):
        if full:
            length = self.full_length
        else:
            length = self.length
        shape = tf.shape(x)
        left_shape = tf.concat([shape[:-1], tf.constant([self.kmin])], axis=0)
        left_pad = tf.zeros(left_shape, dtype=tf.complex64)
        if length == self.kmax:
            x = tf.concat([left_pad, x], axis=2)
        else:
            right_shape = tf.concat([shape[:-1], tf.constant([length - self.kmax])], axis=0)
            right_pad = tf.zeros(right_shape, dtype=tf.complex64)
            x = tf.concat([left_pad, x, right_pad], axis=2)
        return x

    def sigma(self, x, psd):
        inner = 4. * self.delta_f * (tf.math.abs(x) ** 2.) / psd
        sigmasq = tf.math.reduce_sum(inner, axis=2, keepdims=True)
        return real_to_complex(sigmasq ** 0.5)
    

class MatchedFilter(BaseFilter):

    @add_temp_axis
    def matched_filter(self, temp, data, psd, sigma=None, idx=None):

        data = self.cut(data)
        temp = self.cut(temp)
        psd = self.cut(psd)

        if sigma is None:
            sigma = self.sigma(temp, psd)
        lgc = tf.math.not_equal(sigma, tf.zeros_like(sigma))
        sigma = tf.where(lgc, sigma, tf.ones_like(sigma))
        norm_temp = temp / sigma

        ow_data = data / real_to_complex(psd)

        norm_temp = self.pad(norm_temp)
        ow_data = self.pad(ow_data)

        if idx is None:
            return self.matched_filter_ow(norm_temp, ow_data), lgc
        else:
            return self.matched_filter_ow_idx(norm_temp, ow_data, idx), lgc

    @tf.function
    def matched_filter_ow(self, norm_temp, ow_data):

        data = self.cut(ow_data)
        temp = self.cut(norm_temp)

        snr_tilde = 4. * self.delta_f * data * tf.math.conj(temp)

        snr_tilde = self.pad(snr_tilde, full=True)
        snr = tf.signal.ifft(snr_tilde) * self.full_length
        
        return tf.math.abs(snr)

    @tf.function
    def matched_filter_ow_idx(self, norm_temp, ow_data, idx):
        
        data = self.cut(ow_data)
        temp = self.cut(norm_temp)

        snr_tilde = 4. * self.delta_f * data * tf.math.conj(temp)

        snr_tilde = self.pad(snr_tilde, full=True)

        idx = tf.cast(idx, tf.float32)
        idx = tf.expand_dims(idx, axis=1)

        idxs = tf.range(0., self.full_length, delta=1., dtype=tf.float32)
        idxs = tf.expand_dims(idxs, axis=0)

        N = tf.constant(self.full_length, dtype=tf.float32)

        shifter = tf.complex(tf.math.cos(2. * np.pi * idx * idxs / N),
                             tf.math.sin(2. * np.pi * idx * idxs / N))
        shifter = tf.expand_dims(shifter, axis=1)

        snr = tf.math.reduce_sum(snr_tilde * shifter, axis=2)
        return tf.math.abs(snr)


class BaseTransform(BaseFilter):

    @tf.function
    @add_temp_axis
    def get_ortho(self, temp, base, psd,
                  thresh=tf.constant(0.99, dtype=tf.float32)):

        temp = self.cut(temp)
        base = self.cut(base)
        psd = self.cut(psd)

        temp_sigma = self.sigma(temp, psd)
        base_sigma = self.sigma(base, psd)

        temp_norm = temp / temp_sigma
        base_norm = base / base_sigma

        inner = 4 * self.delta_f * tf.math.conj(base_norm) * temp_norm / real_to_complex(psd)
        overlap_cplx = tf.math.reduce_sum(inner, axis=2, keepdims=True)
        overlap = tf.math.abs(overlap_cplx)
        
        thresh = tf.ones_like(overlap) * thresh
        lgc = tf.less(overlap, thresh)
        overlap = tf.where(lgc, overlap_cplx, tf.zeros_like(overlap_cplx))

        ortho = ((temp_norm - overlap * base_norm)
                 / (1 - overlap * tf.math.conj(overlap)) ** 0.5)
        ortho = self.pad(ortho)

        return ortho, lgc


class ShiftTransform(BaseTransform):

    def shift_dt(self, temp, dt):

        dt = tf.expand_dims(dt, 2)

        freqs = tf.expand_dims(self.freqs, 0)
        freqs = tf.expand_dims(freqs, 1)

        shifter = tf.complex(tf.math.cos(- 2. * np.pi * freqs * dt),
                             tf.math.sin(- 2. * np.pi * freqs * dt))

        if tf.rank(temp) == 2:
            temp = tf.expand_dims(temp, 1)

        return temp * shifter

    @tf.custom_gradient
    def shift_df(self, temp, df):

        dj = -1. * tf.math.floordiv(df, self.delta_f)
        dj = tf.cast(dj, tf.int32)
        dj = tf.expand_dims(dj, axis=2)

        if tf.rank(temp) == 2:
            temp = tf.expand_dims(temp, 1)
            temp = tf.repeat(temp, tf.shape(dj)[1], axis=1)

        temp_idxs = tf.range(tf.shape(temp)[1], dtype=tf.int32)
        temp_idxs = tf.expand_dims(temp_idxs, axis=0)
        temp_idxs = tf.repeat(temp_idxs, tf.shape(temp)[0], axis=0)
        temp_idxs = tf.expand_dims(temp_idxs, axis=2)
        temp_idxs = tf.repeat(temp_idxs, tf.shape(temp)[2], axis=2)
        
        batch_idxs = tf.range(tf.shape(temp)[0], dtype=tf.int32)
        batch_idxs = tf.expand_dims(batch_idxs, axis=1)
        batch_idxs = tf.repeat(batch_idxs, tf.shape(temp)[1], axis=1)
        batch_idxs = tf.expand_dims(batch_idxs, axis=2)
        batch_idxs = tf.repeat(batch_idxs, tf.shape(temp)[2], axis=2)

        dj_idxs = tf.range(tf.shape(temp)[2], dtype=tf.int32)
        dj_idxs = tf.expand_dims(dj_idxs, axis=0)
        dj_idxs = tf.expand_dims(dj_idxs, axis=1)
        dj_idxs_f = dj_idxs + dj
        dj_idxs_r = dj_idxs - dj

        dj_idxs_f_low = tf.math.greater_equal(dj_idxs_f, tf.zeros_like(dj_idxs_f))
        dj_idxs_f_high = tf.math.less(dj_idxs_f, tf.ones_like(dj_idxs_f) * tf.shape(temp)[2])
        dj_idxs_f_within = tf.math.logical_and(dj_idxs_f_low, dj_idxs_f_high)
        dj_idxs_f_mask = tf.cast(dj_idxs_f_within, tf.complex64)

        dj_idxs_r_low = tf.math.greater_equal(dj_idxs_r, tf.zeros_like(dj_idxs_r))
        dj_idxs_r_high = tf.math.less(dj_idxs_r, tf.ones_like(dj_idxs_r) * tf.shape(temp)[2])
        dj_idxs_r_within = tf.math.logical_and(dj_idxs_r_low, dj_idxs_r_high)
        dj_idxs_r_mask = tf.cast(dj_idxs_r_within, tf.complex64)

        dj_idxs_f = tf.clip_by_value(dj_idxs_f, 0, tf.shape(temp)[2] - 1)
        dj_idxs_f = tf.stack([batch_idxs, temp_idxs, dj_idxs_f], axis=3)

        dj_idxs_r = tf.clip_by_value(dj_idxs_r, 0, tf.shape(temp)[2] - 1)
        dj_idxs_r = tf.stack([batch_idxs, temp_idxs, dj_idxs_r], axis=3)

        temp = tf.gather_nd(temp, dj_idxs_f) * dj_idxs_f_mask

        def grad(dl_ds):
            # rate of change of shifted template wrt df is equal to -1 times
            # the rate of change of the shifted template wrt f
            ds_df = (temp[:, :, :-2] - temp[:, :, 2:]) / self.delta_f / 2.
            dl_df_real = tf.math.real(dl_ds[:, :, 1:-1]) * tf.math.real(ds_df)
            dl_df_imag = tf.math.imag(dl_ds[:, :, 1:-1]) * tf.math.imag(ds_df)
            dl_df = dl_df_real + dl_df_imag
            dl_df = tf.reduce_sum(dl_df, axis=2)

            ds = tf.gather_nd(dl_ds, dj_idxs_r) * dj_idxs_r_mask
            return ds, dl_df
        
        return temp, grad

    def transform(self, temp, param):
        dt = self.get_dt(param)
        temp = self.shift_dt(temp, dt)
        df = self.get_df(param)
        temp = self.shift_df(temp, df)
        return temp


class PolyShiftTransform(ShiftTransform):

    def __init__(self, nshifts, degree, dt, df, param_max,
                 freqs, f_low, f_high, offset=None):

        super().__init__(freqs, f_low, f_high)

        if degree < 0:
            raise ValueError("degree must be >= 0")
        self.nshifts = nshifts
        self.degree = degree

        self.dt = dt
        self.df = df
        self.param_max = param_max
        
        if offset is None:
            offset = np.pi / nshifts

        dt_base = np.stack([1. * np.sin(2. * np.pi * np.arange(nshifts) / nshifts + offset)
                            for i in range(degree + 1)], axis=-1)
        df_base = np.stack([1. * np.cos(2. * np.pi * np.arange(nshifts) / nshifts + offset)
                            for i in range(degree + 1)], axis=-1)

        self.dt_base = tf.Variable(dt_base, trainable=True, dtype=tf.float32)
        self.df_base = tf.Variable(df_base, trainable=True, dtype=tf.float32)

        self.trainable_weights = {'dt': self.dt_base, 'df': self.df_base}

    def get_dt(self, param):
        dts = tf.expand_dims(self.dt_base, 0)

        param = tf.expand_dims(param, 1)
        param = tf.expand_dims(param, 2)

        param_max = tf.ones_like(param) * self.param_max

        power = tf.range(0, self.degree + 1, delta=1., dtype=tf.float32)
        power = tf.expand_dims(power, 0)
        power = tf.expand_dims(power, 1)
        
        params = tf.math.pow(param, power)
        params_max = tf.math.pow(param_max, power)
        scale = self.dt * params / params_max / (self.degree + 1)

        terms = dts * scale
        dt = tf.math.reduce_sum(terms, axis=2)

        return dt

    def get_df(self, param):
        dfs = tf.expand_dims(self.df_base, 0)

        param = tf.expand_dims(param, 1)
        param = tf.expand_dims(param, 2)

        param_max = tf.ones_like(param) * self.param_max

        power = tf.range(0, self.degree + 1, delta=1., dtype=tf.float32)
        power = tf.expand_dims(power, 0)
        power = tf.expand_dims(power, 1)
        
        params = tf.math.pow(param, power)
        params_max = tf.math.pow(param_max, power)
        scale = self.df * params / params_max / (self.degree + 1)

        terms = dfs * scale
        df = tf.math.reduce_sum(terms, axis=2)

        return df

    def get_weights(self):
        
        power = tf.range(0, self.degree + 1, delta=1., dtype=tf.float32)
        
        param_max = tf.ones_like(power) * self.param_max
        param_max = tf.math.pow(param_max, power)
        scale = 1. / param_max / (self.degree + 1)

        weights = {'dt': self.dt_base * self.dt * scale,
                   'df': self.df_base * self.df * scale}
        return weights

    @classmethod
    def from_weights(cls, dt_weights, df_weights,
                     dt, df, param_max,
                     freqs, f_low, f_high, offset=None):

        nshifts = dt_weights.shape[0]
        degree = dt_weights.shape[1] - 1

        obj = cls(nshifts, degree, dt, df, param_max,
                  freqs, f_low, f_high, offset=offset)

        dt_weights = tf.convert_to_tensor(dt_weights, dtype=tf.float32)
        df_weights = tf.convert_to_tensor(df_weights, dtype=tf.float32)

        power = tf.range(0, obj.degree + 1, delta=1., dtype=tf.float32)
        
        param_max = tf.ones_like(power) * obj.param_max
        param_max = tf.math.pow(param_max, power)
        scale = 1. / param_max / (obj.degree + 1)
        scale = tf.expand_dims(scale, axis=0)

        obj.dt_base = tf.Variable(dt_weights / obj.dt / scale, trainable=True, dtype=tf.float32)
        obj.df_base = tf.Variable(df_weights / obj.df / scale, trainable=True, dtype=tf.float32)

        obj.trainable_weights = {'dt': obj.dt_base, 'df': obj.df_base}

        return obj


class Convolution1DTransform(BaseTransform):

    def __init__(self, nkernel, max_df, max_dt, freqs, f_low, f_high):
        
        super().__init__(freqs, f_low, f_high)

        self.half_width = int(np.ceil(max_df / self.delta_f))

        def normalise(x):
            x = tf.maximum(x, tf.zeros_like(x))
            return x / (tf.math.reduce_sum(x) + 1e-7)

        kernels_real = np.random.randn(self.half_width * 2 + 1, 1, nkernel)
        kernels_real = kernels_real ** 2. / (np.sum(kernels_real ** 2.) + 1e-7)
        self.kernels_real = tf.Variable(kernels_real, dtype=tf.float32,
                                        trainable=True, constraint=normalise)

        kernels_imag = np.random.randn(self.half_width * 2 + 1, 1, nkernel)
        kernels_imag = kernels_imag ** 2. / (np.sum(kernels_imag ** 2.) + 1e-7)
        self.kernels_imag = tf.Variable(kernels_imag, dtype=tf.float32,
                                        trainable=True, constraint=normalise)

        def clip(x):
            x = tf.clip_by_value(x, - max_dt, max_dt)
            return x

        dts = np.random.rand(nkernel) * 2. * max_dt - max_dt
        self.dts = tf.Variable(dts, dtype=tf.float32, trainable=True, constraint=clip)

        self.trainable_weights = {'kernel_real': self.kernels_real,
                                  'kernel_imag': self.kernels_imag,
                                  'dt': self.dts}

    def shift_dt(self, temp):

        dt = tf.expand_dims(self.dts, 0)
        dt = tf.expand_dims(dt, 2)

        freqs = tf.expand_dims(self.freqs, 0)
        freqs = tf.expand_dims(freqs, 1)

        shifter = tf.complex(tf.math.cos(- 2. * np.pi * freqs * dt),
                             tf.math.sin(- 2. * np.pi * freqs * dt))

        if tf.rank(temp) == 2:
            temp = tf.expand_dims(temp, 1)
        
        return temp * shifter

    def convolve(self, temp):

        temp = tf.expand_dims(temp, 1)

        shape = tf.shape(temp)
        pad_shape = tf.concat([shape[:-1], tf.constant([self.half_width])], axis=0)
        pad = tf.zeros(pad_shape, dtype=tf.complex64)
        temp = tf.concat([pad, temp, pad], axis=2)

        temp = tf.transpose(temp, perm=[0, 2, 1])

        temp_real = tf.math.real(temp)
        temp_imag = tf.math.imag(temp)

        ctemp_real = tf.nn.conv1d(temp_real, self.kernels_real, 1, 'VALID', data_format='NWC')
        ctemp_real -= tf.nn.conv1d(temp_imag, self.kernels_imag, 1, 'VALID', data_format='NWC')
        ctemp_imag = tf.nn.conv1d(temp_imag, self.kernels_real, 1, 'VALID', data_format='NWC')
        ctemp_imag += tf.nn.conv1d(temp_real, self.kernels_imag, 1, 'VALID', data_format='NWC')

        ctemp = tf.complex(ctemp_real, ctemp_imag)
        ctemp = tf.transpose(ctemp, perm=[0, 2, 1])

        return ctemp

    @tf.function
    def transform(self, temp):
        temp = self.convolve(temp)
        temp = self.shift_dt(temp)
        return temp


class ChisqFilter(MatchedFilter):

    def __init__(self, transform, threshold, freqs, f_low, f_high):
        super().__init__(freqs, f_low, f_high)
        self.transform = transform
        self.threshold = threshold

    def get_max_idx(self, data, gather_idxs=None):
        if gather_idxs is not None:
            data = tf.gather_nd(data, gather_idxs)
        max_data_idx = tf.math.argmax(data, axis=-1)

        max_data_gather = tf.stack([tf.range(tf.shape(max_data_idx)[0], dtype=tf.int64),
                                    max_data_idx], axis=-1)

        if gather_idxs is not None:
            max_data_idx = max_data_idx + gather_idxs[:, 0, 1]

        max_data = tf.gather_nd(data, max_data_gather)
        return max_data, max_data_idx

    def get_max_snr(self, temp, segs, psds, gather_idxs=None):
        snr, _ = self.matched_filter(temp, segs, psds)
        snr = snr[:, 0, :]
        max_snr, max_snr_idx = self.get_max_idx(snr, gather_idxs=gather_idxs)
        return max_snr, max_snr_idx

    def get_max_snr_prime(self, temp, segs, psds, param,
                          gather_idxs=None, max_snr=False):
        chi_temps = self.transform.transform(temp, param)
        logging.info("Templates transformed")

        chi_orthos, ortho_lgc = self.transform.get_ortho(chi_temps, temp, psds)
        logging.info("Orthogonal templates created")

        snr, _ = self.matched_filter(temp, segs, psds)
        snr = snr[:, 0, :]
        snr_idx = None
        if max_snr:
            snr, snr_idx = self.get_max_snr(temp, segs, psds, gather_idxs=gather_idxs)
        elif gather_idxs is not None:
            snr = tf.gather_nd(snr, gather_idxs)
    
        chis, match_lgc = self.matched_filter(chi_orthos, segs, psds, idx=snr_idx)
        if (not max_snr) and (gather_idxs is not None):
            chis = tf.gather_nd(chis, gather_idxs)
        
        lgc = tf.math.logical_and(ortho_lgc, match_lgc)
        mask = tf.cast(lgc, tf.float32)[:, :, 0]
        ortho_num = tf.reduce_sum(mask, axis=1)

        chis = chis * tf.stop_gradient(mask)
        logging.info("SNRs calculated")

        chisq = tf.math.reduce_sum(chis ** 2., axis=1)
        rchisq = chisq / 2. / tf.stop_gradient(ortho_num)

        chisq_thresh = rchisq / self.threshold
        chisq_thresh = tf.math.maximum(chisq_thresh, tf.ones_like(chisq_thresh))

        snr_prime = snr / chisq_thresh ** 0.5
        if not max_snr:
            snr_prime, _ = self.get_max_idx(snr_prime)
        logging.info("SNR' calculated")

        return snr_prime, chisq_thresh
