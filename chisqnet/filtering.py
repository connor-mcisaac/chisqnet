import logging, h5py, configparser
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


@tf.custom_gradient
def print_gradient(x, name):
    def grad(upstream):
        print(name)
        print(upstream)
        return upstream, None
    return x, grad


@tf.custom_gradient
def divide_gradient(x, y):
    def grad(upstream):
        return upstream / y, None
    return x, grad


@tf.custom_gradient
def no_grad_div(self, x, y):
    def grad(upstream):
        return upstream, None
    return x / y, grad


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

        shift_temp = tf.gather_nd(temp, dj_idxs_f) * dj_idxs_f_mask

        def grad(dl_ds):
            # rate of change of shifted template wrt df is equal to -1 times
            # the rate of change of the shifted template wrt f
            ds_df = (shift_temp[:, :, :-2] - shift_temp[:, :, 2:]) / self.delta_f / 2.
            dl_df_real = tf.math.real(dl_ds[:, :, 1:-1]) * tf.math.real(ds_df)
            dl_df_imag = tf.math.imag(dl_ds[:, :, 1:-1]) * tf.math.imag(ds_df)
            dl_df = dl_df_real + dl_df_imag
            dl_df = tf.reduce_sum(dl_df, axis=2)

            dl_dt = tf.gather_nd(dl_ds, dj_idxs_r) * dj_idxs_r_mask
            return dl_dt, dl_df
        
        return shift_temp, grad

    def transform(self, temp, sample, training=False):
        dt, df = self.get_dt_df(sample, training=training)
        temp = self.shift_df(temp, df)
        temp = self.shift_dt(temp, dt)
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
            sc = ax.scatter(dts[:, i], dfs[:, i], c=params, cmap="cool",
                            alpha=0.5, s=15.)

        if title:
            ax.set_title(title, fontsize="large")

        ax.set_xlim((-self.dt_base * (self.degree + 1),
                     self.dt_base * (self.degree + 1)))
        ax.set_ylim((-self.df_base * (self.degree + 1),
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

        shapes = [len(params)] + layer_sizes + [2]
        self.weights = []
        self.biases = []
        self.trainable_weights = {}
        for i in range(nshifts):
            self.weights.append([])
            self.biases.append([])
            for j in range(len(shapes) - 1):
                weight = tf.random.truncated_normal(
                    (shapes[j], shapes[j + 1]),
                    stddev=tf.math.sqrt(2 / (shapes[j] + shapes[j + 1])),
                    dtype=tf.float32
                )
                self.weights[i] += [tf.Variable(weight, trainable=True, dtype=tf.float32)]
                self.trainable_weights['net_{0}_weight_{1}'.format(i, j)] = self.weights[i][j]

                bias = tf.random.truncated_normal(
                    (shapes[j + 1],),
                    stddev=0.1,
                    dtype=tf.float32
                )
                self.biases[i] += [tf.Variable(bias, trainable=True, dtype=tf.float32)]
                self.trainable_weights['net_{0}_bias_{1}'.format(i, j)] = self.biases[i][j]

    def get_dt_df(self, sample, training=False):
        params = [sample[p] / pb for p, pb in zip(self.params, self.params_base)]
        params = np.stack(params, axis=-1)

        dts = []
        dfs = []
        for ws, bs in zip(self.weights, self.biases):
            values = tf.convert_to_tensor(params[:].astype(np.float32), dtype=tf.float32)
            for w, b in zip(ws[:-1], bs[:-1]):
                values = tf.matmul(values, w) + b
                values = tf.nn.relu(values)
                if training:
                    values = tf.nn.dropout(values, rate=0.5)

            values = tf.matmul(values, ws[-1]) + bs[-1]
            values = tf.math.tanh(values)

            dt_mag, df_mag = tf.split(values, 2, axis=-1)
            dts += [dt_mag * self.dt_max]
            dfs += [df_mag * self.df_max]

        dt = tf.concat(dts, axis=-1)
        df = tf.concat(dfs, axis=-1)
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
                for j in range(len(self.weights[i])):
                    _ = g.create_dataset("net_{0}_weight_{1}".format(i, j),
                                         data=self.weights[i][j].numpy())
                    _ = g.create_dataset("net_{0}_bias_{1}".format(i, j),
                                         data=self.biases[i][j].numpy())

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
            weights = [[g["net_{0}_weight_{1}".format(i, j)][:] for j in range(num)]
                       for i in range(nshifts)]
            biases = [[g["net_{0}_bias_{1}".format(i, j)][:] for j in range(num)]
                      for i in range(nshifts)]

        if isinstance(params[0], bytes):
            params = [p.decode() for p in params]

        obj = cls(nshifts, [], dt_max, df_max, params, params_base,
                  freqs, f_low, f_high)

        obj.weights = [[tf.Variable(w, trainable=True, dtype=tf.float32) for w in ws]
                       for ws in weights]
        obj.biases = [[tf.Variable(b, trainable=True, dtype=tf.float32) for b in bs]
                      for bs in biases]

        obj.trainable_weights = {}
        for i in range(nshifts):
            for j in range(num):
                obj.trainable_weights['net_{0}_weight_{1}'.format(i, j)] = obj.weights[i][j]
                obj.trainable_weights['net_{0}_bias_{1}'.format(i, j)] = obj.biases[i][j]

        return obj

    def plot_model(self, bank, title=None):
        params = {p: bank.table[p] / pb for p, pb in zip(self.params, self.params_base)}
        
        mtotal_min = np.min(bank.table.mtotal)
        mtotal_max = np.max(bank.table.mtotal)
        mass1_min = np.min(bank.table.mass1)
        mass1_max = np.max(bank.table.mass1)
        mass2_min = np.min(bank.table.mass2)
        mass2_max = np.max(bank.table.mass2)
        
        fig, ax = plt.subplots(ncols=5, nrows=2, figsize=(48, 16))

        for i in range(-2, 3):
            params = {'spin1z': np.ones((100,), dtype=np.float32) * (i / 2.),
                      'spin2z': np.ones((100,), dtype=np.float32) * (i / 2.)}

            mass = np.linspace(mtotal_min, mtotal_max, num=100, endpoint=True)
            params['mass1'] = mass / 2.
            params['mass2'] = mass / 2.

            dts, dfs = self.get_dt_df(params)
            dts = dts.numpy()
            dfs = dfs.numpy()

            for j in range(self.nshifts):
                sc = ax[0, i+2].scatter(dts[:, j], dfs[:, j], c=mass, cmap="cool",
                                        alpha=0.5, s=15.)

                ax[0, i+2].set_xlim((-self.dt_max, self.dt_max))
                ax[0, i+2].set_ylim((-self.df_max, self.df_max))

                ax[0, i+2].set_xlabel('Time Shift (s)', fontsize='large')
                ax[0, i+2].set_ylabel('Frequency Shift (Hz)', fontsize='large')

            ax[0, i+2].grid()
            cbar = fig.colorbar(sc, ax=ax[0, i+2])
            cbar.ax.set_ylabel('Total Mass', fontsize='large')

            params['mass1'] = np.linspace(mass1_min, mass1_max, num=100, endpoint=True)
            params['mass2'] = np.ones((100,), dtype=np.float32) * mass2_min

            dts, dfs = self.get_dt_df(params)
            dts = dts.numpy()
            dfs = dfs.numpy()

            for j in range(self.nshifts):
                sc = ax[1, i+2].scatter(dts[:, j], dfs[:, j], c=params['mass1'], cmap="cool",
                                        alpha=0.5, s=15.)

                ax[1, i+2].set_xlim((-self.dt_max, self.dt_max))
                ax[1, i+2].set_ylim((-self.df_max, self.df_max))

                ax[1, i+2].set_xlabel('Time Shift (s)', fontsize='large')
                ax[1, i+2].set_ylabel('Frequency Shift (Hz)', fontsize='large')

            ax[1, i+2].grid()
            cbar = fig.colorbar(sc, ax=ax[1, i+2])
            cbar.ax.set_ylabel('Mass1', fontsize='large')

        if title:
            fig.suptitle(title, fontsize="large")

        return fig, ax


class Convolution1DTransform(BaseTransform):

    def normalise(self, x):
        denom = real_to_complex(tf.math.reduce_mean(tf.math.abs(x), axis=0) + 1e-7)
        return x / denom

    def clip(self, x):
        return tf.clip_by_value(x, - self.max_dt, self.max_dt)

    def __init__(self, nkernel, kernel_df, max_df, max_dt, freqs, f_low, f_high):
        
        super().__init__(freqs, f_low, f_high)

        self.nkernel = nkernel
        self.kernel_df = kernel_df
        self.max_df = max_df
        self.max_dt = max_dt

        self.conv_half_width = int(max_df // self.delta_f)
        conv_freqs = np.arange(self.conv_half_width * 2 + 1) * self.delta_f
        conv_freqs -= conv_freqs[self.conv_half_width + 1]
        self.conv_freqs = tf.convert_to_tensor(conv_freqs.astype(np.float32), dtype=tf.float32)

        self.kernel_half_width = int(max_df // kernel_df)
        self.kernel_freqs = np.arange(self.kernel_half_width * 2 + 1) * self.kernel_df
        self.kernel_freqs -= self.kernel_freqs[self.kernel_half_width + 1]

        diff_f = conv_freqs[:, np.newaxis] - self.kernel_freqs[np.newaxis, :]
        frac = np.maximum(diff_f / self.kernel_df, 0.)

        interp_idx = np.maximum(0, np.sum(diff_f >= 0, axis=1) - 1)
        interp_frac = frac[np.arange(len(conv_freqs)), interp_idx]

        interp_idx = np.repeat(interp_idx[np.newaxis, :, np.newaxis, np.newaxis], nkernel, axis=3)
        kernel_idx = np.repeat(
            np.arange(nkernel)[np.newaxis, np.newaxis, np.newaxis, :],
            self.conv_half_width * 2 + 1, axis=1
        )
        interp_gather = np.stack([interp_idx, kernel_idx], axis=-1)

        interp_frac = np.repeat(interp_frac[np.newaxis, :, np.newaxis, np.newaxis], nkernel, axis=3)

        self.interp_gather = tf.convert_to_tensor(interp_gather, dtype=tf.int64)
        self.interp_frac = tf.convert_to_tensor(interp_frac, dtype=tf.complex64)

        self.kernel_max_df = tf.constant(-1. * self.kernel_freqs[0], dtype=tf.float32)

        kernels_real = tf.random.truncated_normal((self.kernel_half_width * 2 + 1, nkernel),
                                                  dtype=tf.float32)
        kernels_imag = tf.random.truncated_normal((self.kernel_half_width * 2 + 1, nkernel),
                                                  dtype=tf.float32)
        kernels = tf.complex(kernels_real, kernels_imag)
        self.kernels = tf.Variable(self.normalise(kernels),
                                   dtype=tf.complex64, trainable=True,
                                   constraint=self.normalise)

        dts = tf.random.truncated_normal((nkernel,), dtype=tf.float32)
        self.dt_weights = tf.Variable(dts, dtype=tf.float32, trainable=True)

        self.trainable_weights = {'kernels': self.kernels,
                                  'dt_weights': self.dt_weights}

    @tf.function
    def shift_dt(self, temp):
        dt = tf.math.tanh(self.dt_weights) * self.max_dt

        dt = tf.expand_dims(dt, 0)
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

        kernels = self.kernels

        diff = tf.concat([kernels[1:, :] - kernels[:-1, :],
                          tf.zeros([1, self.nkernel], dtype=tf.complex64)], 0)

        kernels = (
            tf.gather_nd(kernels, self.interp_gather)
            + self.interp_frac * tf.gather_nd(diff, self.interp_gather)
        )
        kernels_real = tf.math.real(kernels)
        kernels_imag = tf.math.imag(kernels)

        temp = tf.transpose(temp, perm=[0, 2, 1])
        temp = tf.expand_dims(temp, axis=1)
        temp_real = tf.math.real(temp)
        temp_imag = tf.math.imag(temp)

        padding = [[0, 0], [0, 0], [self.conv_half_width, self.conv_half_width], [0, 0]]

        ctemp_real = tf.nn.conv2d(temp_real, kernels_real, 1, padding, data_format='NHWC')
        ctemp_real -= tf.nn.conv2d(temp_imag, kernels_imag, 1, padding, data_format='NHWC')
        ctemp_imag = tf.nn.conv2d(temp_imag, kernels_real, 1, padding, data_format='NHWC')
        ctemp_imag += tf.nn.conv2d(temp_real, kernels_imag, 1, padding, data_format='NHWC')

        ctemp = tf.complex(ctemp_real, ctemp_imag)
        ctemp = tf.squeeze(ctemp)
        ctemp = tf.transpose(ctemp, perm=[0, 2, 1])

        div = tf.convert_to_tensor(self.conv_half_width * 2 + 1, dtype=tf.complex64)
        ctemp = ctemp / div
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
        kernel_df = config.getfloat(section, 'delta-f')
        max_df = config.getfloat(section, 'frequency-width')
        max_dt = config.getfloat(section, 'max-time-shift')

        obj = cls(nkernel, kernel_df, max_df, max_dt, freqs, f_low, f_high)
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
            _ = g.create_dataset('kernel_df', np.array([self.kernel_df]))
            _ = g.create_dataset('max_df', np.array([self.max_df]))
            _ = g.create_dataset('max_dt', np.array([self.max_dt]))
            _ = g.create_dataset('kernels_real', self.kernels_real.numpy())
            _ = g.create_dataset('kernels_imag', self.kernels_imag.numpy())
            _ = g.create_dataset('dt_weights', self.dt_weights.numpy())

    @classmethod
    def from_file(cls, file_path, freqs, f_low, f_high, group=None):
        with h5py.File(file_path, 'r') as f:
            if group:
                g = f[group]
            else:
                g = f
            nkernel = g['nkernel'][0]
            kernel_df = g['kernel_df'][0]
            max_df = f['max_df'][0]
            max_dt = g['max_dt'][0]
            kernels_real = g['kernels_real'][:]
            kernels_imag = g['kernels_imag'][:]
            dt_weights = g['dt_weights'][:]
            
        obj = cls(nkernel, kernel_df, max_df, max_dt, freqs, f_low, f_high)

        obj.kernels_real = tf.Variable(kernels_real, dtype=tf.float32,
                                       trainable=True, constraint=self.normalise)
        obj.kernels_imag = tf.Variable(kernels_imag, dtype=tf.float32,
                                       trainable=True, constraint=self.normalise)

        obj.dt_weights = tf.Variable(dt_weights, dtype=tf.float32, trainable=True)

        obj.trainable_weights = {'kernel_real': obj.kernels_real,
                                 'kernel_imag': obj.kernels_imag,
                                 'dt': obj.dt_weights}
        return obj

    def plot_model(self, bank, title=None):
        fig, ax = plt.subplots(nrows=self.nkernel,
                               figsize=(8, 6 * self.nkernel))

        kernels_real = tf.math.real(self.kernels)
        kernels_imag = tf.math.imag(self.kernels)

        dt = tf.math.tanh(self.dt_weights) * self.max_dt

        for i in range(self.nkernel):
            ax[i].plot(self.kernel_freqs, kernels_real.numpy()[:, i], label='real', alpha=0.75)
            ax[i].plot(self.kernel_freqs, kernels_imag.numpy()[:, i], label='imag', alpha=0.75)
            
            max_amp = max(np.max(np.abs(kernels_real.numpy()[:, i])),
                          np.max(np.abs(kernels_imag.numpy()[:, i])))
            ax[i].set_ylim([-max_amp, max_amp])

            ax[i].legend()
            ax[i].set_title('Time Shift (s) = {0}'.format(dt[i]), fontsize='small')
            ax[i].set_xlabel('Frequency (Hz)', fontsize='small')
            ax[i].set_ylabel('Amplitude', fontsize='small')
            ax[i].grid()

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

    raise ValueError("{0} is not a valid transformation, select from {1}".format(transformation_key, options.keys()))

