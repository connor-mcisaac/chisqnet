import logging, h5py, configparser
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

    def transform(self, temp, params, training=False):

        err = "This method shoulkd be overwritten by a child class. "
        err += "Implement this method before using this class."
        raise NotImplementedError(err)

    @classmethod
    def from_config(cls, config_file, freqs, f_low, f_high, section="model"):

        err = "This method shoulkd be overwritten by a child class. "
        err += "Implement this method before using this class."
        raise NotImplementedError(err)

    def to_file(self, file_path, group=None, append=False):

        err = "This method shoulkd be overwritten by a child class. "
        err += "Implement this method before using this class."
        raise NotImplementedError(err)

    @classmethod
    def from_file(cls, file_path, freqs, f_low, f_high, group=None):

        err = "This method shoulkd be overwritten by a child class. "
        err += "Implement this method before using this class."
        raise NotImplementedError(err)


class ShiftTransform(BaseTransform):

    @tf.function
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

    def transform(self, temp, sample, training=False):
        dt, df = self.get_dt_df(sample, training=training)
        temp = self.shift_dt(temp, dt)
        temp = self.shift_df(temp, df)
        return temp


class PolyShiftTransform(ShiftTransform):

    def __init__(self, nshifts, degree, dt_base, df_base, param, param_base,
                 freqs, f_low, f_high, offset=None):

        super().__init__(freqs, f_low, f_high)

        if degree < 0:
            raise ValueError("degree must be >= 0")
        self.nshifts = nshifts
        self.degree = degree

        self.dt_base = dt_base
        self.df_base = df_base

        self.param = param
        self.param_base = param_base
        
        if offset is None:
            offset = np.pi / nshifts

        dt_shift = np.zeros((nshifts, degree + 1), dtype=np.float32)
        df_shift = np.zeros((nshifts, degree + 1), dtype=np.float32)

        dt_shift[:, 0] = np.sin(2. * np.pi * np.arange(nshifts) / nshifts + offset)
        df_shift[:, 0] = np.cos(2. * np.pi * np.arange(nshifts) / nshifts + offset)

        self.dt_shift = tf.Variable(dt_shift, trainable=True, dtype=tf.float32)
        self.df_shift = tf.Variable(df_shift, trainable=True, dtype=tf.float32)

        self.trainable_weights = {'dt': self.dt_shift, 'df': self.df_shift}

    def get_dt_df(self, sample, training=False):
        param = tf.convert_to_tensor(sample[self.param][:].astype(np.float32),
                                     dtype=tf.float32)

        dts = tf.expand_dims(self.dt_shift, 0)
        dfs = tf.expand_dims(self.df_shift, 0)

        param = tf.expand_dims(param, 1)
        param = tf.expand_dims(param, 2)

        param_base = tf.ones_like(param) * self.param_base

        power = tf.range(0, self.degree + 1, delta=1., dtype=tf.float32)
        power = tf.expand_dims(power, 0)
        power = tf.expand_dims(power, 1)
        
        params = tf.math.pow(param, power)
        params_base = tf.math.pow(param_base, power)
        
        dt_scale = self.dt_base * params / params_base
        df_scale = self.df_base * params / params_base

        dt_terms = dts * dt_scale
        dt = tf.math.reduce_sum(dt_terms, axis=2)

        df_terms = dfs * df_scale
        df = tf.math.reduce_sum(df_terms, axis=2)

        return dt, df

    @classmethod
    def from_config(cls, config_file, freqs, f_low, f_high, section="model"):
        config = configparser.ConfigParser()
        config.read(config_file)
        
        nshifts = config.getint(section, "shift-num")
        degree = config.getint(section, "poly-degree")
        dt_base = config.getfloat(section, "base-time-shift")
        df_base = config.getfloat(section, "base-freq-shift")
        param = config.get(section, "shift-param")
        param_base = config.getfloat(section, "shift-param-base")
        offset = config.getfloat(section, "offset", fallback=None)

        obj = cls(nshifts, degree, dt_base, df_base, param, param_base,
                  freqs, f_low, f_high, offset=offset)
        return obj

    def to_file(self, file_path, group=None, append=False):
        if append:
            file_mode = 'a'
        else:
            file_mode = 'w'

        with h5py.File(file_path, file_mode) as f:
            if group:
                g = f.create_group(group)
            else:
                g = f
            _ = g.create_dataset("nshifts", data=np.array([self.nshifts]))
            _ = g.create_dataset("degree", data=np.array([self.degree]))
            _ = g.create_dataset("dt_base", data=np.array([self.dt_base]))
            _ = g.create_dataset("df_base", data=np.array([self.df_base]))
            _ = g.create_dataset("param", data=np.array([self.param]).astype('S'))
            _ = g.create_dataset("param_base", data=np.array([self.param_base]))
            _ = g.create_dataset("dt_shift", data=self.dt_shift.numpy())
            _ = g.create_dataset("df_shift", data=self.df_shift.numpy())
        
    @classmethod
    def from_file(cls, file_path, freqs, f_low, f_high, group=None):
        with h5py.File(file_path, 'r') as f:
            if group:
                g = f[group]
            else:
                g = f
            nshifts = g["nshifts"][0]
            degree = g["degree"][0]
            dt_base = g["dt_base"][0]
            df_base = g["df_base"][0]
            param = g["param"][0]
            param_base = g["param_base"][0]
            dt_shift = g["dt_shift"][:]
            df_shift = g["df_shift"][:]

        if isinstance(param, bytes):
            param = param.decode()

        obj = cls(nshifts, degree, dt_base, df_base, param, param_base,
                  freqs, f_low, f_high)

        obj.dt_shift = tf.Variable(dt_shift, trainable=True, dtype=tf.float32)
        obj.df_shift = tf.Variable(df_shift, trainable=True, dtype=tf.float32)

        obj.trainable_weights = {'dt': obj.dt_shift, 'df': obj.df_shift}

        return obj

    def plot_model(self, bank, title=None):
        param = bank.table[self.param]
        params = np.linspace(np.min(param), np.max(param), num=100, endpoint=True)
        
        dts, dfs = self.get_dt_df({self.param: params})
        dts = dts.numpy()
        dfs = dfs.numpy()

        fig, ax = plt.subplots(figsize=(8, 6))

        for i in range(self.nshifts):
            sc = ax.scatter(dts[:, i], dfs[:, i], c=params, cmap="cool", alpha=0.5)

        if title:
            ax.set_title(title, fontsize="large")

        ax.set_xlim((-self.dt_base * (self.degree + 1),
                     self.dt_base * (self.degree + 1)))
        ax.set_xlim((-self.df_base * (self.degree + 1),
                     self.df_base * (self.degree + 1)))

        ax.set_xlabel('Time Shift (s)', fontsize='large')
        ax.set_ylabel('Frequency Shift (Hz)', fontsize='large')
        ax.grid()

        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel(self.param, fontsize='large')

        return fig, ax


class NetShiftTransform(ShiftTransform):

    def __init__(self, nshifts, layer_sizes, dt_max, df_max, params, params_base,
                 freqs, f_low, f_high):

        super().__init__(freqs, f_low, f_high)

        self.nshifts = nshifts

        self.dt_max = dt_max
        self.df_max = df_max

        if isinstance(params, list):
            self.params = params
        else:
            self.params = [params]

        if isinstance(params_base, list):
            self.params_base = params_base
        else:
            self.params_base = [params_base]

        shapes = [len(params)] + layer_sizes + [nshifts * 2]
        self.weights = []
        self.biases = []
        self.trainable_weights = {}
        for i in range(len(shapes) - 1):
            weight = tf.random.truncated_normal((shapes[i], shapes[i + 1]),
                                                stddev=tf.math.sqrt(2 / (shapes[i] + shapes[i + 1])),
                                                dtype=tf.float32)
            self.weights += [tf.Variable(weight, trainable=True, dtype=tf.float32)]
            self.trainable_weights['weight_{0}'.format(i)] = self.weights[-1]
            if i != (len(shapes) - 2):
                bias = tf.zeros((shapes[i + 1]), dtype=tf.float32)
                self.biases += [tf.Variable(bias, trainable=True, dtype=tf.float32)]
                self.trainable_weights['bias_{0}'.format(i)] = self.biases[-1]

    def get_dt_df(self, sample, training=False):
        params = [sample[p] / pb for p, pb in zip(self.params, self.params_base)]
        params = np.stack(params, axis=-1)
        values = tf.convert_to_tensor(params.astype(np.float32), dtype=tf.float32)

        for w, b in zip(self.weights[:-1], self.biases):
            values = tf.matmul(values, w) + b
            values = tf.math.tanh(values)
            if training:
                values += tf.random.normal(tf.shape(values), stddev=0.01,
                                           dtype=tf.float32)

        values = tf.matmul(values, self.weights[-1])
        values = tf.math.tanh(values)
        if training:
            values += tf.random.normal(tf.shape(values), stddev=0.01,
                                       dtype=tf.float32)

        dt_mag, df_mag = tf.split(values, 2, axis=-1)
        dt = dt_mag * self.dt_max
        df = df_mag * self.df_max

        return dt, df

    @classmethod
    def from_config(cls, config_file, freqs, f_low, f_high, section="model"):
        config = configparser.ConfigParser()
        config.read(config_file)
        
        nshifts = config.getint(section, "shift-num")
        layer_sizes = [int(l) for l in config.get(section, "layer-sizes").split(',')]
        dt_max = config.getfloat(section, "max-time-shift")
        df_max = config.getfloat(section, "max-freq-shift")
        params = config.get(section, "shift-params").split(',')
        params_base = [float(p) for p in config.get(section, "shift-params-base").split(',')]

        obj = cls(nshifts, layer_sizes, dt_max, df_max, params, params_base,
                  freqs, f_low, f_high)
        return obj

    def to_file(self, file_path, group=None, append=False):
        if append:
            file_mode = 'a'
        else:
            file_mode = 'w'

        with h5py.File(file_path, file_mode) as f:
            if group:
                g = f.create_group(group)
            else:
                g = f
            _ = g.create_dataset("nshifts", data=np.array([self.nshifts]))
            _ = g.create_dataset("dt_max", data=np.array([self.dt_max]))
            _ = g.create_dataset("df_max", data=np.array([self.df_max]))
            _ = g.create_dataset("params", data=np.array(self.params).astype('S'))
            _ = g.create_dataset("params_base", data=np.array(self.params_base))
            g.attrs['layers_num'] = len(self.weights)
            for i in range(len(self.weights)):
                _ = g.create_dataset("weights_{0}".format(i), data=self.weights[i].numpy())
                if i != (len(self.weights) - 1):
                    _ = g.create_dataset("biases_{0}".format(i), data=self.biases[i].numpy())

    @classmethod
    def from_file(cls, file_path, freqs, f_low, f_high, group=None):
        with h5py.File(file_path, 'r') as f:
            if group:
                g = f[group]
            else:
                g = f
            nshifts = g["nshifts"][0]
            dt_max = g["dt_max"][0]
            df_max = g["df_max"][0]
            params = list(g["params"][:])
            params_base = list(g["params_base"][:])
            num = g.attrs['layers_num']
            weights = [g["weights_{0}".format(i)][:] for i in range(num)]
            biases = [g["biases_{0}".format(i)][:] for i in range(num)]

        if isinstance(params[0], bytes):
            params = [p.decode() for p in params]

        obj = cls(nshifts, [], dt_max, df_max, params, params_base,
                  freqs, f_low, f_high)

        obj.weights = [tf.Variable(w, trainable=True, dtype=tf.float32) for w in weights]
        obj.biases = [tf.Variable(b, trainable=True, dtype=tf.float32) for b in biases]

        obj.trainable_weights = {}
        for i in range(num):
            obj.trainable_weights['weight_{0}'.format(i)] = obj.weights[i]
            if i != (num - 1):
                obj.trainable_weights['bias_{0}'.format(i)] = obj.biases[i]

        return obj

    def plot_model(self, bank, title=None):
        params = {p: bank.table[p] / pb for p, pb in zip(self.params, self.params_base)}
        
        dts, dfs = self.get_dt_df(params)
        dts = dts.numpy()
        dfs = dfs.numpy()

        mass1 = bank.table.mass1
        mass2 = bank.table.mass2
        
        duration = bank.table.template_duration
        spin = bank.table.chi_eff

        fig, ax = plt.subplots(ncols=self.nshifts + 1, nrows=4,
                               figsize=(6 * (self.nshifts + 1), 18))
        
        for i in range(self.nshifts):
            sct = ax[0, i+1].scatter(mass1, mass2, c=dts[:, i],
                                     vmin=-self.dt_max, vmax=self.dt_max,
                                     cmap="coolwarm")
            sct = ax[1, i+1].scatter(duration, spin, c=dts[:, i],
                                     vmin=-self.dt_max, vmax=self.dt_max,
                                     cmap="coolwarm")
            scf = ax[2, i+1].scatter(mass1, mass2, c=dfs[:, i],
                                     vmin=-self.df_max, vmax=self.df_max,
                                     cmap="coolwarm")
            scf = ax[3, i+1].scatter(duration, spin, c=dfs[:, i],
                                     vmin=-self.df_max, vmax=self.df_max,
                                     cmap="coolwarm")
            ax[0, i+1].set_xscale('log')
            ax[0, i+1].set_yscale('log')
            ax[1, i+1].set_xscale('log')
            ax[1, i+1].set_yscale('log')
            ax[2, i+1].set_xscale('log')
            ax[2, i+1].set_yscale('log')
            ax[3, i+1].set_xscale('log')
            ax[3, i+1].set_yscale('log')
            ax[0, i+1].grid()
            ax[1, i+1].grid()
            ax[2, i+1].grid()
            ax[3, i+1].grid()
        
        cbt = fig.colorbar(sct, ax=ax[0:2, 0], location='left')
        cbt.ax.set_ylabel('Time Shift (s)', fontsize='small')
        cbf = fig.colorbar(scf, ax=ax[2:4, 0], location='left')
        cbf.ax.set_ylabel('Frequency Shift (Hz)', fontsize='small')
        ax[0, 0].axis('off')
        ax[1, 0].axis('off')
        ax[2, 0].axis('off')
        ax[3, 0].axis('off')
        if title:
            fig.suptitle(title, fontsize="large")

        return fig, ax


class Convolution1DTransform(BaseTransform):

    def __init__(self, nkernel, max_df, max_dt, freqs, f_low, f_high):
        
        super().__init__(freqs, f_low, f_high)

        self.nkernel = nkernel
        self.max_df = max_df
        self.max_dt = max_dt
        self.half_width = int(max_df // self.delta_f)

        def normalise(x):
            return x / (tf.math.reduce_sum(tf.math.abs(x)) + 1e-7)

        kernels_real = tf.random.truncated_normal((self.half_width * 2 + 1, 1, nkernel),
                                                  dtype=tf.float32)
        kernels_real = normalise(kernels_real)
        self.kernels_real = tf.Variable(kernels_real, dtype=tf.float32,
                                        trainable=True, constraint=normalise)

        kernels_imag = tf.random.truncated_normal((self.half_width * 2 + 1, 1, nkernel),
                                                  dtype=tf.float32)
        kernels_imag = normalise(kernels_imag)
        self.kernels_imag = tf.Variable(kernels_imag, dtype=tf.float32,
                                        trainable=True, constraint=normalise)

        def clip(x):
            x = tf.clip_by_value(x, - max_dt, max_dt)
            return x

        dts = np.zeros(nkernel)
        self.dts = tf.Variable(dts, dtype=tf.float32, trainable=True, constraint=clip)

        self.trainable_weights = {'kernel_real': self.kernels_real,
                                  'kernel_imag': self.kernels_imag,
                                  'dt': self.dts}

    @tf.function
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

    @tf.function
    def convolve(self, temp, training=False):

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


    def transform(self, temp, sample, training=False):
        temp = self.convolve(temp, training=training)
        temp = self.shift_dt(temp)
        return temp

    @classmethod
    def from_config(cls, config_file, freqs, f_low, f_high, section="model"):
        config = configparser.ConfigParser()
        config.read(config_file)

        nkernel = config.getint(section, 'kernel-num')
        max_df = config.getfloat(section, 'frequency-width')
        max_dt = config.getfloat(section, 'max-time-shift')

        obj = cls(nkernel, max_df, max_dt, freqs, f_low, f_high)
        return obj

    def to_file(self, file_path, group=None, append=False):
        if append:
            file_mode = 'a'
        else:
            file_mode = 'w'

        with h5py.File(file_path, file_mode) as f:
            if group:
                g = f.create_group(group)
            else:
                g = f
            _ = g.create_dataset('nkernel', np.array([self.nkernel]))
            _ = g.create_dataset('max_df', np.array([self.max_df]))
            _ = g.create_dataset('max_dt', np.array([self.max_dt]))
            _ = g.create_dataset('kernels_real', self.kernels_real.numpy())
            _ = g.create_dataset('kernels_imag', self.kernels_imag.numpy())
            _ = g.create_dataset('dts', self.dts.numpy())

    @classmethod
    def from_file(cls, file_path, freqs, f_low, f_high, group=None):
        with h5py.File(file_path, 'r') as f:
            if group:
                g = f[group]
            else:
                g = f
            nkernel = g['nkernel'][0]
            max_df = f['max_df'][0]
            max_dt = g['max_dt'][0]
            kernels_real = g['kernels_real'][:]
            kernels_imag = g['kernels_imag'][:]
            dts = g['dts'][:]
            
        obj = cls(nkernel, max_df, max_dt, freqs, f_low, f_high)

        def normalise(x):
            return x / (tf.math.reduce_sum(tf.math.abs(x)) + 1e-7)

        obj.kernels_real = tf.Variable(kernels_real, dtype=tf.float32,
                                       trainable=True, constraint=normalise)
        obj.kernels_imag = tf.Variable(kernels_imag, dtype=tf.float32,
                                       trainable=True, constraint=normalise)

        def clip(x):
            x = tf.clip_by_value(x, - max_dt, max_dt)
            return x

        obj.dts = tf.Variable(dts, dtype=tf.float32, trainable=True, constraint=clip)

        obj.trainable_weights = {'kernel_real': obj.kernels_real,
                                 'kernel_imag': obj.kernels_imag,
                                 'dt': obj.dts}
        return obj

    def plot_model(self, bank, title=None):
        freqs = np.arange(self.half_width * 2 + 1) * self.delta_f
        freqs -= freqs[self.half_width + 1]

        fig, ax = plt.subplots(nrows=self.nkernel,
                               figsize=(8, 6 * self.nkernel))

        for i in range(self.nkernel):
            ax[i].plot(freqs, self.kernels_real.numpy()[:, 0, i], label='real', alpha=0.75)
            ax[i].plot(freqs, self.kernels_imag.numpy()[:, 0, i], label='imag', alpha=0.75)
            
            max_amp = max(np.max(np.abs(self.kernels_real.numpy()[:, 0, i])),
                          np.max(np.abs(self.kernels_imag.numpy()[:, 0, i])))
            ax[i].set_ylim([-max_amp, max_amp])

            ax[i].legend()
            ax[i].set_title('Time Shift (s) = {0}'.format(self.dts[i]), fontsize='small')
            ax[i].set_xlabel('Frequency (Hz)', fontsize='small')
            ax[i].set_ylabel('Amplitude', fontsize='small')

        if title:
            fig.suptitle(title, fontsize="large")

        return fig, ax


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

    def get_max_snr_prime(self, temp, segs, psds, params,
                          gather_idxs=None, max_snr=False, **kwargs):
        chi_temps = self.transform.transform(temp, params, **kwargs)
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

        lgc = tf.math.logical_and(ortho_lgc, match_lgc)
        mask = tf.cast(lgc, tf.float32)
        if max_snr:
            mask = mask[:, :, 0]
        ortho_num = tf.reduce_sum(mask, axis=1)
        ortho_num = tf.math.maximum(ortho_num, tf.ones_like(ortho_num))

        chis = chis * tf.stop_gradient(mask)
        logging.info("SNRs calculated")

        chisq = tf.math.reduce_sum(chis ** 2., axis=1)

        if (not max_snr) and (gather_idxs is not None):
            chisq = tf.gather_nd(chisq, gather_idxs)

        rchisq = chisq / 2. / tf.stop_gradient(ortho_num)

        chisq_thresh = rchisq / self.threshold
        chisq_thresh = tf.math.maximum(chisq_thresh, tf.ones_like(chisq_thresh))

        snr_prime = snr / chisq_thresh ** 0.5

        if not max_snr:
            snr_prime, max_idx = self.get_max_idx(snr_prime)
            gather_max = tf.stack([tf.range(len(max_idx), dtype=tf.int64), max_idx], axis=-1)
            chisq_thresh = tf.gather_nd(chisq_thresh, gather_max)
        logging.info("SNR' calculated")
        return snr_prime, chisq_thresh


def select_transformation(transformation_key):
    options = {
        'polyshift': PolyShiftTransform,
        'netshift': NetShiftTransform,
        'convolution': Convolution1DTransform
    }

    if transformation_key in options.keys():
        return options[transformation_key]

    raise ValueError("{0} is not a valid transformation, select from {1}".format(options.keys()))

