import logging, h5py, configparser
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from pycbc.types import FrequencySeries
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
def no_grad_div(x, y):
    def grad(upstream):
        return upstream, None
    return x / y, grad


@tf.custom_gradient
def no_grad_mul(x, y):
    def grad(upstream):
        return upstream, None
    return x * y, grad


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


def create_interpolator(fin, fout, temp_num=1):
    delta = (fin[1] - fin[0])
    mask = (fout >= fin[0]) & (fout <= fin[-1])
    mask = mask[np.newaxis, np.newaxis, :].astype(np.float32)
    interp_idx = np.clip(np.floor_divide(fout - fin[0], delta), 0., len(fin) - 1)
    interp_frac = np.maximum(np.mod(fout - fin[0], delta) / delta, 0.)
    
    interp_idx = np.repeat(interp_idx[np.newaxis, np.newaxis, :], temp_num, axis=1)
    temp_idx = np.repeat(np.arange(temp_num)[np.newaxis, :, np.newaxis], len(fout), axis=2)
    interp_gather = np.stack([temp_idx, interp_idx], axis=-1)
    
    interp_frac = np.repeat(interp_frac[np.newaxis, np.newaxis, :], temp_num, axis=1)
    
    mask = tf.convert_to_tensor(mask, dtype=tf.float32)
    interp_gather = tf.convert_to_tensor(interp_gather, dtype=tf.int64)
    interp_frac = tf.convert_to_tensor(interp_frac, dtype=tf.float32)
        
    def _interpolate(temp):
        t_shape = tf.shape(temp)
        g_shape = tf.shape(interp_gather)
        
        batch_idx = tf.range(t_shape[0], dtype=tf.int64)
        batch_idx = tf.expand_dims(batch_idx, axis=1)
        batch_idx = tf.repeat(batch_idx, g_shape[1], axis=1)
        batch_idx = tf.expand_dims(batch_idx, axis=2)
        batch_idx = tf.repeat(batch_idx, g_shape[2], axis=2)
        batch_idx = tf.expand_dims(batch_idx, axis=3)
        
        gather = tf.repeat(interp_gather, t_shape[0], axis=0)
        gather = tf.concat([batch_idx, gather], axis=3)
        
        diff = tf.concat([temp[:, :, 1:] - temp[:, :, :-1],
                          tf.zeros([t_shape[0], t_shape[1], 1], dtype=tf.float32)],
                         axis=2)
        interp = (
            tf.gather_nd(temp, gather)
            + interp_frac * tf.gather_nd(diff, gather)
        )
        return interp * mask
        
    return _interpolate


def create_complex_interpolator(fin, fout, temp_num=1):
    interpolator = create_interpolator(fin, fout, temp_num=temp_num)
    
    def _interpolator(temp):
        temp_abs = tf.math.abs(temp)
        temp_phase = tf.math.angle(temp)

        temp_phase_diff = temp_phase[:, :, 1:] - temp_phase[:, :, :-1]

        lgc = tf.math.greater(tf.math.abs(temp_phase_diff),
                              tf.ones_like(temp_phase_diff) * np.pi)
        mask = tf.cast(lgc, tf.float32)
        sign = tf.math.sign(temp_phase_diff)

        mod = - 1. * sign * mask * 2. * np.pi
        cummod = tf.math.cumsum(mod, axis=2)
        shape = tf.shape(temp)
        cummod = tf.concat([tf.zeros((shape[0], shape[1], 1), dtype=tf.float32), cummod],
                           axis=2)
        temp_phase = temp_phase + tf.stop_gradient(cummod)

        temp_abs = interpolator(temp_abs)
        temp_phase = interpolator(temp_phase)

        temp = tf.complex(temp_abs * tf.math.cos(temp_phase),
                          temp_abs * tf.math.sin(temp_phase))
        return temp

    return _interpolator


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
        mask = tf.cast(lgc, tf.complex64)

        sigma = tf.where(lgc, sigma, tf.ones_like(sigma))
        norm_temp = mask * temp / sigma

        ow_data = data / real_to_complex(psd)

        norm_temp = self.pad(norm_temp)
        ow_data = self.pad(ow_data)

        if idx is None:
            return self.matched_filter_ow(norm_temp, ow_data)
        else:
            return self.matched_filter_ow_idx(norm_temp, ow_data, idx)

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

        snr = tf.math.reduce_sum(snr_tilde * shifter, axis=2, keepdims=True)
        return tf.math.abs(snr)


class BaseTransform(BaseFilter):
    transformation_type = "base"

    def __init__(self, freqs, f_low, f_high, l1_reg=0., l2_reg=0.):
        super().__init__(freqs, f_low, f_high)
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.trainable_weights = {}

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
        overlap_abs = tf.math.abs(overlap_cplx)

        thresh = tf.ones_like(overlap_abs) * thresh
        lgc = tf.less(overlap_abs, thresh)
        mask = tf.cast(lgc, tf.complex64)
        overlap_cplx = tf.where(lgc, overlap_cplx, tf.zeros_like(overlap_cplx))

        ortho = ((temp_norm - overlap_cplx * base_norm)
                 / (1 - overlap_cplx * tf.math.conj(overlap_cplx)) ** 0.5)
        ortho = self.pad(ortho) * mask
        return ortho, lgc

    def transform(self, temp, params, training=False):

        err = "This method should be overwritten by a child class. "
        err += "Implement this method before using this class."
        raise NotImplementedError(err)

    def pycbc_transform(self, temp, params):
        temp = self.transform(temp, params, training=False)
        return temp

    @classmethod
    def from_config(cls, config_file, freqs, f_low, f_high, section="model"):

        err = "This method should be overwritten by a child class. "
        err += "Implement this method before using this class."
        raise NotImplementedError(err)

    def to_file(self, file_path, group=None, append=False):

        err = "This method should be overwritten by a child class. "
        err += "Implement this method before using this class."
        raise NotImplementedError(err)

    @classmethod
    def from_file(cls, file_path, freqs, f_low, f_high, group=None):

        err = "This method should be overwritten by a child class. "
        err += "Implement this method before using this class."
        raise NotImplementedError(err)

    def get_regulariser_loss(self):
        losses = []
        for n, w in self.trainable_weights.items():
            l1_loss = 0.
            l2_loss = 0.
            if self.l1_reg:
                l1_loss = self.l1_reg * tf.reduce_sum(tf.math.abs(w))
            if self.l2_reg:
                l2_loss = self.l2_reg * tf.reduce_sum(tf.math.square(w))
            losses.append(l1_loss + l2_loss)
        loss = tf.math.add_n(losses)
        return loss


class TemplateMixer(object):

    def __init__(self, templates_in, templates_out, layer_sizes,
                 params, params_base, l1_reg=0., l2_reg=0.):
        self.templates_in = templates_in
        self.templates_out = templates_out

        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        
        if isinstance(params, list):
            self.params = params
        else:
            self.params = [params]

        if isinstance(params_base, list):
            self.params_base = params_base
        else:
            self.params_base = [params_base]

        shapes = [len(params)] + layer_sizes + [templates_in]
        self.weights = []
        self.biases = []
        self.trainable_weights = {}

        for i in range(templates_out):
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

                bias = tf.zeros((shapes[j + 1],), dtype=tf.float32)
                self.biases[i] += [tf.Variable(bias, trainable=True, dtype=tf.float32)]
                self.trainable_weights['net_{0}_bias_{1}'.format(i, j)] = self.biases[i][j]

    def get_mixer(self, sample, training=False):
        params = [sample[p] / pb for p, pb in zip(self.params, self.params_base)]
        params = np.stack(params, axis=-1)

        mixes = []
        for ws, bs in zip(self.weights, self.biases):
            values = tf.convert_to_tensor(params[:].astype(np.float32), dtype=tf.float32)
            for w, b in zip(ws[:-1], bs[:-1]):
                values = tf.matmul(values, w) + b
                values = tf.nn.relu(values)
            values = tf.matmul(values, ws[-1]) + bs[-1]
            values = 1e-6 + (1 - 1e-6) * tf.math.sigmoid(values)
            if training:
                noise = tf.random.truncated_normal(tf.shape(values), stddev=0.1)
                values = values + noise
            values = values / tf.reduce_sum(values, axis=1, keepdims=True)
            mixes += [values]

        mixer = tf.stack(mixes, axis=-1)
        return mixer

    def mix_temps(self, temps, sample, training=False):
        mixer = self.get_mixer(sample, training=training)
        mixer = real_to_complex(mixer)

        temps = tf.transpose(temps, perm=[0, 2, 1])
        temps = tf.matmul(temps, mixer)
        temps = tf.transpose(temps, perm=[0, 2, 1])

        return temps

    @classmethod
    def from_config(cls, config_file, section="model"):
        config = configparser.ConfigParser()
        config.read(config_file)

        templates_in = config.getint(section, 'templates-in')
        templates_out = config.getint(section, 'templates-out')
        layer_sizes = [int(l) for l in config.get(section, "layer-sizes").split(',')]
        params = config.get(section, "params").split(',')
        params_base = [float(p) for p in config.get(section, "params-base").split(',')]
        l1_reg = config.getfloat(section, "l1-regulariser", fallback=0.)
        l2_reg = config.getfloat(section, "l2-regulariser", fallback=0.)


        obj = cls(templates_in, templates_out, layer_sizes,
                  params, params_base, l1_reg=l1_reg, l2_reg=l2_reg)
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
            _ = g.create_dataset('templates_out', data=np.array([self.templates_out]))
            _ = g.create_dataset('params', data=np.array(self.params).astype('S'))
            _ = g.create_dataset('params_base', data=np.array(self.params_base))
            g.attrs['layers_num'] = len(self.weights[0])
            for i in range(len(self.weights)):
                for j in range(len(self.weights[i])):
                    _ = g.create_dataset("net_{0}_weight_{1}".format(i, j),
                                         data=self.weights[i][j].numpy())
                    _ = g.create_dataset("net_{0}_bias_{1}".format(i, j),
                                         data=self.biases[i][j].numpy())
            _ = g.create_dataset("l1_reg", data=np.array([self.l1_reg]))
            _ = g.create_dataset("l2_reg", data=np.array([self.l2_reg]))

    @classmethod
    def from_file(cls, file_path, group=None):
        with h5py.File(file_path, 'r') as f:
            if group:
                g = f[group]
            else:
                g = f
            templates_out = g['templates_out'][0]
            params = list(g["params"][:])
            params_base = list(g["params_base"][:])
            num = g.attrs['layers_num']
            weights = [[g["net_{0}_weight_{1}".format(i, j)][:] for j in range(num)]
                       for i in range(templates_out)]
            biases = [[g["net_{0}_bias_{1}".format(i, j)][:] for j in range(num)]
                      for i in range(templates_out)]
            l1_reg = g['l1_reg'][0]
            l2_reg = g['l2_reg'][0]

        if isinstance(params[0], bytes):
            params = [p.decode() for p in params]

        obj = cls(1, templates_out, [],
                  params, params_base, l1_reg=l1_reg, l2_reg=l2_reg)

        obj.weights = [[tf.Variable(w, trainable=True, dtype=tf.float32) for w in ws]
                       for ws in weights]
        obj.biases = [[tf.Variable(b, trainable=True, dtype=tf.float32) for b in bs]
                      for bs in biases]

        obj.trainable_weights = {}
        for i in range(templates_out):
            for j in range(num):
                obj.trainable_weights['net_{0}_weight_{1}'.format(i, j)] = obj.weights[i][j]
                obj.trainable_weights['net_{0}_bias_{1}'.format(i, j)] = obj.biases[i][j]

        return obj

    def plot_model(self, bank, title=None):
        params = {p: bank.table[p] / pb for p, pb in zip(self.params, self.params_base)}
        
        mixer = self.get_mixer(params)
        mixer = mixer.numpy()

        mtotal = bank.table.mtotal
        duration = bank.table.template_duration

        fig, ax = plt.subplots(nrows=self.templates_in, ncols=self.templates_out * 2,
                               figsize=(16 * self.templates_out, 6 * self.templates_in))

        for i in range(self.templates_in):
            for j in range(self.templates_out):
                sct = ax[i, j*2].scatter(mtotal, mixer[:, i, j], c=mixer[:, i, j],
                                         vmin=0, vmax=1,
                                         cmap="inferno")
                sct = ax[i, j*2+1].scatter(duration, mixer[:, i, j], c=mixer[:, i, j],
                                           vmin=0, vmax=1,
                                           cmap="inferno")
                ax[i, j*2].set_ylim((0, 1))
                ax[i, j*2+1].set_ylim((0, 1))
                ax[i, j*2].set_xscale('log')
                ax[i, j*2+1].set_xscale('log')
                ax[i, j*2].grid()
                ax[i, j*2+1].grid()

        if title:
            fig.suptitle(title, fontsize="large")

        return fig, ax

    def get_regulariser_loss(self):
        losses = []
        for n, w in self.trainable_weights.items():
            l1_loss = 0.
            l2_loss = 0.
            if self.l1_reg:
                l1_loss = self.l1_reg * tf.reduce_sum(tf.math.abs(w))
            if self.l2_reg:
                l2_loss = self.l2_reg * tf.reduce_sum(tf.math.square(w))
            losses.append(l1_loss + l2_loss)
        loss = tf.math.add_n(losses)
        return loss


class ChisqFilter(MatchedFilter):

    def __init__(self, freqs, f_low, f_high, transform, mixer=None,
                 threshold_max=4., constant_max=2., alpha_max=6., beta_max=6.,
                 train_threshold=False, train_constant=False,
                 train_alpha=False, train_beta=False,
                 l1_reg=0., l2_reg=0.):
        super().__init__(freqs, f_low, f_high)
        self.transform = transform
        self.mixer = mixer
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg
        self.threshold_max = threshold_max
        self.train_threshold = train_threshold
        thresh_const = lambda x: tf.clip_by_value(x, 1., self.threshold_max)
        self.threshold = tf.Variable(np.array([1. + (threshold_max - 1.) / 2.], dtype=np.float32),
                                     trainable=train_threshold,
                                     dtype=tf.float32, constraint=thresh_const)

        self.constant_max = constant_max
        self.train_constant = train_constant
        constant_const = lambda x: tf.clip_by_value(x, 1., self.constant_max)
        self.constant = tf.Variable(np.array([1. + (constant_max - 1.) / 2.], dtype=np.float32),
                                    trainable=train_constant,
                                    dtype=tf.float32, constraint=constant_const)

        self.alpha_max = alpha_max
        self.train_alpha = train_alpha
        alpha_const = lambda x: tf.clip_by_value(x, 1., self.alpha_max)
        self.alpha = tf.Variable(np.array([1. + (alpha_max - 1.) / 2.], dtype=np.float32),
                                 trainable=train_alpha,
                                 dtype=tf.float32, constraint=alpha_const)

        self.beta_max = beta_max
        self.train_beta = train_beta
        beta_const = lambda x: tf.clip_by_value(x, 1., self.beta_max)
        self.beta = tf.Variable(np.array([1. + (beta_max - 1.) / 2.], dtype=np.float32),
                                trainable=train_beta,
                                dtype=tf.float32, constraint=beta_const)

        self.trainable_weights = {}
        for k, v in self.transform.trainable_weights.items():
            self.trainable_weights[k] = v
        if self.mixer:
            for k, v in self.mixer.trainable_weights.items():
                self.trainable_weights[k] = v
        if train_threshold:
            self.trainable_weights['threshold'] = self.threshold
        if train_constant:
            self.trainable_weights['constant'] = self.constant
        if train_alpha:
            self.trainable_weights['alpha'] = self.alpha
        if train_beta:
            self.trainable_weights['beta'] = self.beta

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
        snr = self.matched_filter(temp, segs, psds)
        snr = snr[:, 0, :]
        max_snr, max_snr_idx = self.get_max_idx(snr, gather_idxs=gather_idxs)
        return max_snr, max_snr_idx

    def get_max_snr_prime(self, temp, segs, psds, params,
                          gather_idxs=None, max_snr=False, training=False):

        chi_temps = self.transform.transform(temp, params, training=training)
        logging.info("Templates transformed")

        chi_orthos, ortho_lgc = self.transform.get_ortho(chi_temps, temp, psds)
        logging.info("Orthogonal templates created")

        if self.mixer:
            chi_orthos = self.mixer.mix_temps(chi_orthos, params, training=training)
            chi_orthos, ortho_lgc = self.transform.get_ortho(chi_orthos, temp, psds)

        dof = 2 * tf.shape(chi_orthos)[1]
        dof = tf.cast(dof, tf.float32)

        snr = self.matched_filter(temp, segs, psds)
        snr = snr[:, 0, :]
        snr_idx = None
        if max_snr:
            snr, snr_idx = self.get_max_idx(snr, gather_idxs=gather_idxs)
        elif gather_idxs is not None:
            snr = tf.gather_nd(snr, gather_idxs)

        chis = self.matched_filter(chi_orthos, segs, psds, idx=snr_idx)
        logging.info("SNRs calculated")

        chisq = tf.where(ortho_lgc, chis ** 2., tf.ones_like(chis) * 2.)
        chisq = tf.math.reduce_sum(chisq, axis=1)

        if max_snr:
            chisq = chisq[:, 0]
        elif gather_idxs is not None:
            chisq = tf.gather_nd(chisq, gather_idxs)

        rchisq = chisq / dof
        if training:
            chisq_mod = (rchisq / self.threshold) ** self.beta
            chisq_mod = (chisq_mod + self.constant) / (tf.math.tanh(chisq_mod) + self.constant)
        else:
            rchisq_thresh = tf.math.maximum(rchisq, tf.ones_like(rchisq) * self.threshold)
            chisq_mod = (rchisq_thresh / self.threshold) ** self.beta
            chisq_mod = (chisq_mod + self.constant) / (1. + self.constant)
        snr_prime = snr / chisq_mod ** (1. / self.alpha)

        if not max_snr:
            snr_prime, max_idx = self.get_max_idx(snr_prime)
            gather_max = tf.stack([tf.range(tf.shape(max_idx)[0], dtype=tf.int64), max_idx], axis=-1)
            snr = tf.gather_nd(snr, gather_max)
            rchisq = tf.gather_nd(rchisq, gather_max)
        logging.info("SNR' calculated")
        return snr_prime, snr, rchisq

    def get_max_snr_prime_idx(self, temp, segs, psds, params, idxs, training=False):

        chi_temps = self.transform.transform(temp, params, training=training)
        logging.info("Templates transformed")

        chi_orthos, ortho_lgc = self.transform.get_ortho(chi_temps, temp, psds)
        logging.info("Orthogonal templates created")

        if self.mixer:
            chi_orthos = self.mixer.mix_temps(chi_orthos, params, training=training)
            chi_orthos, ortho_lgc = self.transform.get_ortho(chi_orthos, temp, psds)

        dof = 2 * tf.shape(chi_orthos)[1]
        dof = tf.cast(dof, tf.float32)

        snr = self.matched_filter(temp, segs, psds, idx=idxs)
        snr = snr[:, 0, 0]

        chis = self.matched_filter(chi_orthos, segs, psds, idx=idxs)
        logging.info("SNRs calculated")

        chisq = tf.where(ortho_lgc, chis ** 2., tf.ones_like(chis) * 2.)
        chisq = tf.math.reduce_sum(chisq, axis=1)
        chisq = chisq[:, 0]

        rchisq = chisq / dof
        if training:
            chisq_mod = (rchisq / self.threshold) ** self.beta
            chisq_mod = (chisq_mod + self.constant) / (tf.math.tanh(chisq_mod) + self.constant)
        else:
            rchisq_thresh = tf.math.maximum(rchisq, tf.ones_like(rchisq) * self.threshold)
            chisq_mod = (rchisq_thresh / self.threshold) ** self.beta
            chisq_mod = (chisq_mod + self.constant) / (1. + self.constant)
        snr_prime = snr / chisq_mod ** (1. / self.alpha)

        logging.info("SNR' calculated")
        return snr_prime, snr, rchisq

    @classmethod
    def from_config(cls, config_file, freqs, f_low, f_high):
        config = configparser.ConfigParser()
        config.read(config_file)

        transform_type = config.get("model", "transformation")
        transform_class = select_transformation(transform_type)
        transform = transform_class.from_config(
            config_file, freqs, f_low, f_high, section="transformation"
        )

        if config.has_section("mixer"):
            mixer = TemplateMixer.from_config(config_file, section="mixer")
        else:
            mixer = None

        threshold_max = config.getfloat("model", "threshold-max")
        constant_max = config.getfloat("model", "constant-max")
        alpha_max = config.getfloat("model", "alpha-max")
        beta_max = config.getfloat("model", "beta-max")

        train_threshold = config.getboolean("model", "train-threshold", fallback=False)
        train_constant = config.getboolean("model", "train-constant", fallback=False)
        train_alpha = config.getboolean("model", "train-alpha", fallback=False)
        train_beta = config.getboolean("model", "train-beta", fallback=False)

        l1_reg = config.getfloat("model", "l1-regulariser", fallback=0.)
        l2_reg = config.getfloat("model", "l2-regulariser", fallback=0.)

        obj = cls(freqs, f_low, f_high, transform, mixer=mixer,
                  threshold_max=threshold_max,
                  constant_max=constant_max,
                  alpha_max=alpha_max,
                  beta_max=beta_max,
                  train_threshold=train_threshold,
                  train_constant=train_constant,
                  train_alpha=train_alpha,
                  train_beta=train_beta,
                  l1_reg=l1_reg, l2_reg=l2_reg)
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
            _ = g.create_dataset("threshold_max", data=np.array([self.threshold_max]))
            _ = g.create_dataset("threshold", data=self.threshold.numpy())
            _ = g.create_dataset("train_threshold", data=np.array([int(self.train_threshold)]))
            _ = g.create_dataset("constant_max", data=np.array([self.constant_max]))
            _ = g.create_dataset("constant", data=self.constant.numpy())
            _ = g.create_dataset("train_constant", data=np.array([int(self.train_constant)]))
            _ = g.create_dataset("alpha_max", data=np.array([self.alpha_max]))
            _ = g.create_dataset("alpha", data=self.alpha.numpy())
            _ = g.create_dataset("train_alpha", data=np.array([int(self.train_alpha)]))
            _ = g.create_dataset("beta_max", data=np.array([self.beta_max]))
            _ = g.create_dataset("beta", data=self.beta.numpy())
            _ = g.create_dataset("train_beta", data=np.array([int(self.train_beta)]))
            _ = g.create_dataset("l1_reg", data=np.array([self.l1_reg]))
            _ = g.create_dataset("l2_reg", data=np.array([self.l2_reg]))

        if group is None:
            group = ''
        self.transform.to_file(file_path, group=group + '/transform', append=True)
        if self.mixer:
            self.mixer.to_file(file_path, group=group + '/mixer', append=True)

    @classmethod
    def from_file(cls, file_path, freqs, f_low, f_high, group=None):
        with h5py.File(file_path, 'r') as f:
            if group:
                g = f[group]
            else:
                g = f
            threshold_max = g["threshold_max"][0]
            threshold = g["threshold"][0]
            train_threshold = bool(g['train_threshold'][0])
            constant_max = g["constant_max"][0]
            constant = g["constant"][0]
            train_constant = bool(g['train_constant'][0])
            alpha_max = g["alpha_max"][0]
            alpha = g["alpha"][0]
            train_alpha = bool(g['train_alpha'][0])
            beta_max = g["beta_max"][0]
            beta = g["beta"][0]
            train_beta = bool(g['train_beta'][0])
            l1_reg = g["l1_reg"][0]
            l2_reg = g["l2_reg"][0]

            if 'mixer' in g.keys():
                mixer_check = True
            else:
                mixer_check = False

        if group is None:
            group = ''
        transform = load_transformation(file_path, freqs, f_low, f_high,
                                        group=group + '/transform')
        if mixer_check:
            mixer = TemplateMixer.from_file(file_path, group=group + '/mixer')
        else:
            mixer = None

        obj = cls(freqs, f_low, f_high, transform, mixer=mixer,
                  threshold_max=threshold_max,
                  constant_max=constant_max,
                  alpha_max=alpha_max,
                  beta_max=beta_max,
                  train_threshold=train_threshold,
                  train_constant=train_constant,
                  train_alpha=train_alpha,
                  train_beta=train_beta,
                  l1_reg=l1_reg, l2_reg=l2_reg)

        thresh_const = lambda x: tf.clip_by_value(x, 1., obj.threshold_max)
        obj.threshold = tf.Variable(np.array([threshold], dtype=np.float32),
                                    trainable=train_threshold,
                                    dtype=tf.float32, constraint=thresh_const)

        constant_const = lambda x: tf.clip_by_value(x, 0., obj.constant_max)
        obj.constant = tf.Variable(np.array([constant], dtype=np.float32),
                                   trainable=train_constant,
                                   dtype=tf.float32, constraint=constant_const)

        alpha_const = lambda x: tf.clip_by_value(x, 1., obj.alpha_max)
        obj.alpha = tf.Variable(np.array([alpha], dtype=np.float32),
                                trainable=train_alpha,
                                dtype=tf.float32, constraint=alpha_const)

        beta_const = lambda x: tf.clip_by_value(x, 1., obj.beta_max)
        obj.beta = tf.Variable(np.array([beta], dtype=np.float32),
                               trainable=train_beta,
                               dtype=tf.float32, constraint=beta_const)

        if train_threshold:
            obj.trainable_weights['threshold'] = obj.threshold
        if train_constant:
            obj.trainable_weights['constant'] = obj.constant
        if train_alpha:
            obj.trainable_weights['alpha'] = obj.alpha
        if train_beta:
            obj.trainable_weights['beta'] = obj.beta

        return obj

    def get_regulariser_loss(self):
        losses = [self.transform.get_regulariser_loss()]
        if self.mixer:
            losses += [self.mixer.get_regulariser_loss()]
        if self.train_threshold:
            losses += [self.l1_reg * (self.threshold_max - self.threshold[0])]
            losses += [self.l2_reg * (self.threshold_max - self.threshold[0])]
        if self.train_constant:
            losses += [self.l1_reg * (self.constant_max - self.constant[0])]
            losses += [self.l2_reg * (self.constant_max - self.constant[0])]
        if self.train_alpha:
            losses += [self.l1_reg * (self.alpha_max - self.alpha[0])]
            losses += [self.l2_reg * (self.alpha_max - self.alpha[0])]
        if self.train_beta:
            losses += [self.l1_reg * (self.beta[0] - 1.)]
            losses += [self.l2_reg * (self.beta[0] - 1.)]
        loss = tf.math.add_n(losses)
        return loss

    def plot_model(self, trig_snrs, trig_rchisq, inj_snrs, inj_rchisq, title=None):
        fig, ax = plt.subplots(figsize=(8, 6))

        chi_cont = np.logspace(-2, 4, 300)
        for c in [5, 7, 9, 11, 13]:
            chisq_mod = (((chi_cont / self.threshold.numpy()) ** self.beta.numpy()
                          + self.constant.numpy())
                         / (1. + self.constant.numpy()))
            chisq_mod = np.maximum(chisq_mod, np.ones_like(chisq_mod))

            snr_cont = c * (chisq_mod ** (1. / self.alpha.numpy()))

            ax.plot(snr_cont, chi_cont, color='black', alpha=0.5, linestyle='--')

        ax.scatter(trig_snrs, trig_rchisq, c='black', s=5., marker='o', label='Noise', alpha=0.75)
        ax.scatter(inj_snrs, inj_rchisq, s=20., marker='^', label='Injections', alpha=0.75)

        ax.scatter([], [], label='threshold = {0:.2f}'.format(self.threshold.numpy()[0]), c='w')
        ax.scatter([], [], label='constant = {0:.2f}'.format(self.constant.numpy()[0]), c='w')
        ax.scatter([], [], label='alpha = {0:.2f}'.format(self.alpha.numpy()[0]), c='w')
        ax.scatter([], [], label='beta = {0:.2f}'.format(self.beta.numpy()[0]), c='w')

        ax.legend(loc='upper left')

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.set_xlim((min(trig_snrs.min(), inj_snrs.min()) * 0.9,
                     max(trig_snrs.max(), inj_snrs.max()) * 1.5))
        ax.set_ylim((min(trig_rchisq.min(), inj_rchisq.min()) * 0.9,
                     max(trig_rchisq.max(), inj_rchisq.max()) * 1.5))

        ax.set_xlabel('SNR', fontsize='large')
        ax.set_ylabel('Reduced $\chi^2$', fontsize='large')
        ax.grid()

        if title:
            ax.set_title(title, fontsize="large")

        return fig, ax


class ShiftTransform(BaseTransform):
    transformation_type = "shift"

    @tf.function
    def shift_dt(self, temp, dt):

        dt = tf.expand_dims(dt, 2)

        freqs = tf.expand_dims(self.freqs, 0)
        freqs = tf.expand_dims(freqs, 1)

        shifter = tf.complex(tf.math.cos(- 2. * np.pi * freqs * dt),
                             tf.math.sin(- 2. * np.pi * freqs * dt))

        return temp * shifter

    @tf.function
    def shift_df(self, temp, df):

        mod = tf.math.floormod(df, self.delta_f)
        df = df - tf.stop_gradient(mod)
        df = tf.expand_dims(df, 2)

        shape = tf.shape(temp)
        batch = shape[0]

        temp = tf.signal.irfft(temp)

        times = tf.expand_dims(self.times, 0)
        times = tf.expand_dims(times, 1)

        shifter = tf.complex(tf.math.cos(2. * np.pi * df * times),
                             tf.math.sin(2. * np.pi * df * times))

        if tf.size(tf.shape(temp)) == 2:
            temp = tf.expand_dims(temp, 1)

        temp = real_to_complex(temp)
        temp = temp * shifter
        temp = tf.signal.fft(temp)

        temp = temp[..., :self.length]
        df = df[..., :self.length]

        freqs = tf.expand_dims(self.freqs, 0)
        freqs = tf.expand_dims(freqs, 1)

        low = freqs - self.delta_f / 2. - df
        low = low / tf.math.abs(low)
        low = tf.math.maximum(low, tf.zeros_like(low))
        low = real_to_complex(low)

        high = tf.math.reduce_max(freqs) - freqs - self.delta_f / 2. + df
        high = high / tf.math.abs(high)
        high = tf.math.maximum(high, tf.zeros_like(high))
        high = real_to_complex(high)

        return temp * tf.stop_gradient(low) * tf.stop_gradient(high)

    @tf.custom_gradient
    def fast_shift_df(self, temp, df):

        sign = tf.math.sign(df)
        size = tf.math.abs(df)
        dj = - sign * tf.math.floordiv(size, self.delta_f)
        dj = tf.cast(dj, tf.int32)
        dj = tf.expand_dims(dj, axis=2)

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

        dj_idxs_f = tf.clip_by_value(dj_idxs_f, 0, tf.shape(temp)[2] - 1)
        dj_idxs_f = tf.stack([batch_idxs, temp_idxs, dj_idxs_f], axis=3)

        dj_idxs_r_low = tf.math.greater_equal(dj_idxs_r, tf.zeros_like(dj_idxs_r))
        dj_idxs_r_high = tf.math.less(dj_idxs_r, tf.ones_like(dj_idxs_r) * tf.shape(temp)[2])
        dj_idxs_r_within = tf.math.logical_and(dj_idxs_r_low, dj_idxs_r_high)
        dj_idxs_r_mask = tf.cast(dj_idxs_r_within, tf.complex64)

        dj_idxs_r = tf.clip_by_value(dj_idxs_r, 0, tf.shape(temp)[2] - 1)
        dj_idxs_r = tf.stack([batch_idxs, temp_idxs, dj_idxs_r], axis=3)

        shift_temp = tf.gather_nd(temp, dj_idxs_f) * dj_idxs_f_mask

        def grad(dl_ds):
            delta = int(0.05 / self.delta_f)
            ds_df = shift_temp[:, :, 2*delta:] - shift_temp[:, :, :-2*delta]
            dl_df_real = tf.math.real(dl_ds[:, :, delta:-delta]) * tf.math.real(ds_df)
            dl_df_imag = tf.math.imag(dl_ds[:, :, delta:-delta]) * tf.math.imag(ds_df)
            dl_df = dl_df_real + dl_df_imag
            dl_df = tf.reduce_sum(dl_df, axis=2)

            dl_dt = tf.gather_nd(dl_ds, dj_idxs_r) * dj_idxs_r_mask
            return dl_dt, dl_df
        
        return shift_temp, grad

    def transform(self, temp, sample, training=False):
        dt, df = self.get_dt_df(sample, training=training)
        if tf.rank(temp) == 2:
            temp = tf.expand_dims(temp, 1)
        temp = tf.repeat(temp, tf.shape(df)[1], axis=1)
        temp = self.shift_dt(temp, dt)
        temp = self.fast_shift_df(temp, df)
        return temp


class PolyShiftTransform(ShiftTransform):
    transformation_type = "polyshift"

    def __init__(self, nshifts, degree, dt_base, df_base, param, param_base,
                 freqs, f_low, f_high, offset=None, l1_reg=0., l2_reg=0.):

        super().__init__(freqs, f_low, f_high, l1_reg=l1_reg, l2_reg=l2_reg)

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
        
        dt_scale = params / params_base
        df_scale = params / params_base

        dt_terms = dts * dt_scale
        dt = tf.math.reduce_sum(dt_terms, axis=2)

        df_terms = dfs * df_scale
        df = tf.math.reduce_sum(df_terms, axis=2)

        return dt * self.dt_base, df * self.df_base

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
        l1_reg = config.getfloat(section, "l1-regulariser", fallback=0.)
        l2_reg = config.getfloat(section, "l2-regulariser", fallback=0.)

        obj = cls(nshifts, degree, dt_base, df_base, param, param_base,
                  freqs, f_low, f_high, offset=offset,
                  l1_reg=l1_reg, l2_reg=l2_reg)
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
            g.attrs['transformation_type'] = self.transformation_type
            _ = g.create_dataset("nshifts", data=np.array([self.nshifts]))
            _ = g.create_dataset("degree", data=np.array([self.degree]))
            _ = g.create_dataset("dt_base", data=np.array([self.dt_base]))
            _ = g.create_dataset("df_base", data=np.array([self.df_base]))
            _ = g.create_dataset("param", data=np.array([self.param]).astype('S'))
            _ = g.create_dataset("param_base", data=np.array([self.param_base]))
            _ = g.create_dataset("dt_shift", data=self.dt_shift.numpy())
            _ = g.create_dataset("df_shift", data=self.df_shift.numpy())
            _ = g.create_dataset("l1_reg", data=np.array([self.l1_reg]))
            _ = g.create_dataset("l2_reg", data=np.array([self.l2_reg]))

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
            l1_reg = g["l1_reg"][0]
            l2_reg = g["l2_reg"][0]

        if isinstance(param, bytes):
            param = param.decode()

        obj = cls(nshifts, degree, dt_base, df_base, param, param_base,
                  freqs, f_low, f_high, l1_reg=l1_reg, l2_reg=l2_reg)

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
    transformation_type = "netshift"

    def __init__(self, nshifts, dt_max, df_max, template_df,
                 shared_layers, shift_layers,
                 freqs, f_low, f_high,
                 min_freq=None, max_freq=None,
                 l1_reg=0., l2_reg=0., dropout=0.):
        
        super().__init__(freqs, f_low, f_high, l1_reg=l1_reg, l2_reg=l2_reg)

        self.nshifts = nshifts
        self.dt_max = dt_max
        self.df_max = df_max

        self.template_df = template_df
        self.shared_layers = shared_layers
        self.shift_layers = shift_layers

        if min_freq is None:
            min_freq = f_low
        if max_freq is None:
            max_freq = f_high

        self.min_freq = min_freq
        self.max_freq = max_freq

        self.dropout = dropout

        template_freqs = np.arange(min_freq, max_freq, template_df)
        self.to_dense = create_interpolator(self.freqs.numpy(), template_freqs)

        self.trainable_weights = {}
        self.shared_weights = []
        self.shared_biases = []

        in_layer = len(template_freqs)
        for i, n in enumerate(shared_layers):
            weight = tf.random.truncated_normal(
                (in_layer, n),
                stddev=tf.cast(tf.math.sqrt(2. / (in_layer + n)), tf.float32),
                dtype=tf.float32
            )
            self.shared_weights += [tf.Variable(weight, trainable=True, dtype=tf.float32)]
            self.trainable_weights['shared_weight_{0}'.format(i)] = self.shared_weights[i]
                
            bias = tf.zeros((n,), dtype=tf.float32)
            self.shared_biases += [tf.Variable(bias, trainable=True, dtype=tf.float32)]
            self.trainable_weights['shared_bias_{0}'.format(i)] = self.shared_biases[i]
            
            in_layer = n

        self.shift_weights = []
        self.shift_biases = []

        for i in range(nshifts):
            self.shift_weights.append([])
            self.shift_biases.append([])

            in_layer = shared_layers[-1]
            for j, n in enumerate(shift_layers + [2]):
                weight = tf.random.truncated_normal(
                    (in_layer, n),
                    stddev=tf.cast(tf.math.sqrt(2. / (in_layer + n)), tf.float32),
                    dtype=tf.float32
                )
                self.shift_weights[i] += [tf.Variable(weight, trainable=True, dtype=tf.float32)]
                self.trainable_weights['shift_{0}_weight_{1}'.format(i, j)] = self.shift_weights[i][j]
                
                bias = tf.zeros((n,), dtype=tf.float32)
                self.shift_biases[i] += [tf.Variable(bias, trainable=True, dtype=tf.float32)]
                self.trainable_weights['shift_{0}_bias_{1}'.format(i, j)] = self.shift_biases[i][j]
                in_layer = n

    def dense(self, temp, training=False):
        ntemp = tf.math.abs(temp)
        ntemp = self.to_dense(ntemp)
        ntemp = ntemp / tf.math.reduce_mean(ntemp, axis=2, keepdims=True)

        for i in range(len(self.shared_weights)):
            if i == 0:
                feature = tf.matmul(ntemp, self.shared_weights[i]) + self.shared_biases[i]
            else:
                feature = tf.matmul(feature, self.shared_weights[i]) + self.shared_biases[i]
            feature = tf.nn.relu(feature)
            if training:
                feature = tf.nn.dropout(feature, self.dropout)

        dts = []
        dfs = []
        
        for i in range(self.nshifts):
            for j in range(len(self.shift_weights[i]) - 1):
                if j == 0:
                    value = tf.matmul(feature, self.shift_weights[i][j]) + self.shift_biases[i][j]
                else:
                    value = tf.matmul(value, self.shift_weights[i][j]) + self.shift_biases[i][j]
                value = tf.nn.relu(value)
                if training:
                    value = tf.nn.dropout(value, self.dropout)
            value = tf.matmul(value, self.shift_weights[i][-1]) + self.shift_biases[i][-1]
            value = tf.math.tanh(value)
            dt, df = tf.split(value, 2, axis=2)
            dts += [dt]
            dfs += [df]
        dts = tf.concat(dts, axis=2)
        dfs = tf.concat(dfs, axis=2)

        return dts * self.dt_max, dfs * self.df_max

    def transform(self, temp, sample, training=False):
        if tf.rank(temp) == 2:
            temp = tf.expand_dims(temp, 1)
        dt, df = self.dense(temp, training=training)
        if tf.shape(temp)[1] == 1:
            temp = tf.repeat(temp, tf.shape(df)[2], axis=1)
        temp = self.shift_dt(temp, dt[:, 0, :])
        temp = self.fast_shift_df(temp, df[:, 0, :])
        return temp

    @classmethod
    def from_config(cls, config_file, freqs, f_low, f_high, section="model"):
        config = configparser.ConfigParser()
        config.read(config_file)

        nshifts = config.getint(section, "shift-num")
        dt_max = config.getfloat(section, "max-time-shift")
        df_max = config.getfloat(section, "max-freq-shift")
        template_df = config.getfloat(section, "template-df")
        shared_layers = [int(l) for l in config.get(section, "shared-layers").split(',')]
        shift_layers = [int(l) for l in config.get(section, "shift-layers").split(',')]
        min_freq = config.getfloat(section, "min-freq", fallback=None)
        max_freq = config.getfloat(section, "max-freq", fallback=None)
        l1_reg = config.getfloat(section, "l1-regulariser", fallback=0.)
        l2_reg = config.getfloat(section, "l2-regulariser", fallback=0.)
        dropout = config.getfloat(section, "dropout", fallback=0.)

        obj = cls(nshifts, dt_max, df_max,
                  template_df, shared_layers, shift_layers,
                  freqs, f_low, f_high,
                  min_freq=min_freq, max_freq=max_freq,
                  l1_reg=l1_reg, l2_reg=l2_reg, dropout=dropout)
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
            g.attrs['transformation_type'] = self.transformation_type
            _ = g.create_dataset("nshifts", data=np.array([self.nshifts]))
            _ = g.create_dataset("dt_max", data=np.array([self.dt_max]))
            _ = g.create_dataset("df_max", data=np.array([self.df_max]))
            _ = g.create_dataset('template_df', data=np.array([self.template_df]))
            _ = g.create_dataset('layers', data=np.array(self.layers))
            _ = g.create_dataset('min_freq', data=np.array([self.min_freq]))
            _ = g.create_dataset('max_freq', data=np.array([self.max_freq]))
            _ = g.create_dataset("l1_reg", data=np.array([self.l1_reg]))
            _ = g.create_dataset("l2_reg", data=np.array([self.l2_reg]))
            _ = g.create_dataset("dropout", data=np.array([self.dropout]))
            for i in range(len(self.weights)):
                _ = g.create_dataset("weight_{0}".format(i),
                                     data=self.weights[i].numpy())
                _ = g.create_dataset("bias_{0}".format(i),
                                     data=self.biases[i].numpy())

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
            template_df = g['template_df'][0]
            layers = [int(l) for l in g['layers'][:]]
            min_freq = g['min_freq'][0]
            max_freq = g['max_freq'][0]
            l1_reg = g["l1_reg"][0]
            l2_reg = g["l2_reg"][0]
            dropout = g["dropout"][0]
            weights = []
            biases = []
            means = []
            variances = []
            for i in range(len(layers) + 1):
                weights += [g['weight_{0}'.format(i)][:]]
                biases += [g['bias_{0}'.format(i)][:]]

        obj = cls(nshifts, dt_max, df_max,
                  template_df, layers, freqs, f_low, f_high,
                  min_freq=min_freq, max_freq=max_freq,
                  l1_reg=l1_reg, l2_reg=l2_reg, dropout=dropout)

        obj.weights = []
        obj.biases = []
        obj.trainable_weights = {}
        for i in range(len(layers) + 1):
            weight = tf.convert_to_tensor(weights[i], dtype=tf.float32)
            obj.weights += [tf.Variable(weight, trainable=True, dtype=tf.float32)]
            obj.trainable_weights['weight_{0}'.format(i)] = obj.weights[i]
            bias = tf.convert_to_tensor(biases[i], dtype=tf.float32)
            obj.biases += [tf.Variable(bias, trainable=True, dtype=tf.float32)]
            obj.trainable_weights['bias_{0}'.format(i)] = obj.biases[i]

        return obj

    def plot_model(self, bank, title=None, param='mtotal'):
        params = bank.table[param]
        tids = np.argsort(params)

        dts = []
        dfs = []
        ps = []
        for tid in tids:
            temp = bank[tid].numpy().copy()[np.newaxis, np.newaxis, :]
            temp = tf.convert_to_tensor(temp, dtype=tf.complex64)
            dt, df = self.dense(temp, training=False)
            ps += [params[tid]]
            dts += [dt.numpy()[0, 0, :]]
            dfs += [df.numpy()[0, 0, :]]
        ps = np.array(ps)
        dts = np.stack(dts, axis=0)
        dfs = np.stack(dfs, axis=0)

        fig, ax = plt.subplots(figsize=(8, 6))

        for i in range(self.nshifts):
            sc = ax.scatter(dts[:, i], dfs[:, i], c=ps, cmap="cool",
                            alpha=0.5, s=15.)

        if title:
            ax.set_title(title, fontsize="large")

        ax.set_xlim((-self.dt_max, self.dt_max))
        ax.set_ylim((-self.df_max, self.df_max))

        ax.set_xlabel('Time Shift (s)', fontsize='large')
        ax.set_ylabel('Frequency Shift (Hz)', fontsize='large')
        ax.grid()

        cbar = plt.colorbar(sc)
        cbar.ax.set_ylabel(param, fontsize='large')

        return fig, ax


class QuadNetShiftTransform(NetShiftTransform):
    transformation_type = "quadnetshift"

    def __init__(self, dt_max, df_max, template_df,
                 shared_layers, shift_layers,
                 freqs, f_low, f_high,
                 min_freq=None, max_freq=None,
                 l1_reg=0., l2_reg=0., dropout=0.):
        
        super().__init__(4, dt_max, df_max, template_df,
                         shared_layers, shift_layers,
                         freqs, f_low, f_high,
                         min_freq=min_freq, max_freq=max_freq,
                         l1_reg=l1_reg, l2_reg=l2_reg, dropout=dropout)

    def dense(self, temp, training=False):
        ntemp = tf.math.abs(temp)
        ntemp = self.to_dense(ntemp)
        ntemp = ntemp / tf.math.reduce_mean(ntemp, axis=2, keepdims=True)

        for i in range(len(self.shared_weights)):
            if i == 0:
                feature = tf.matmul(ntemp, self.shared_weights[i]) + self.shared_biases[i]
            else:
                feature = tf.matmul(feature, self.shared_weights[i]) + self.shared_biases[i]
            feature = tf.nn.relu(feature)
            if training:
                feature = tf.nn.dropout(feature, self.dropout)

        dts = []
        dfs = []
        
        for i, (t_sign, f_sign) in enumerate(zip([1., 1., -1., -1.], [1., -1., 1., -1.])):
            for j in range(len(self.shift_weights[i]) - 1):
                if j == 0:
                    value = tf.matmul(feature, self.shift_weights[i][j]) + self.shift_biases[i][j]
                else:
                    value = tf.matmul(value, self.shift_weights[i][j]) + self.shift_biases[i][j]
                value = tf.nn.relu(value)
                if training:
                    value = tf.nn.dropout(value, self.dropout)
            value = tf.matmul(value, self.shift_weights[i][-1]) + self.shift_biases[i][-1]
            value = tf.math.tanh(value) * 0.5
            dt, df = tf.split(value, 2, axis=2)
            dts += [dt + 0.5 * t_sign]
            dfs += [df + 0.5 * f_sign]
        dts = tf.concat(dts, axis=2)
        dfs = tf.concat(dfs, axis=2)

        return dts * self.dt_max, dfs * self.df_max

    @classmethod
    def from_config(cls, config_file, freqs, f_low, f_high, section="model"):
        config = configparser.ConfigParser()
        config.read(config_file)

        dt_max = config.getfloat(section, "max-time-shift")
        df_max = config.getfloat(section, "max-freq-shift")
        template_df = config.getfloat(section, "template-df")
        shared_layers = [int(l) for l in config.get(section, "shared-layers").split(',')]
        shift_layers = [int(l) for l in config.get(section, "shift-layers").split(',')]
        min_freq = config.getfloat(section, "min-freq", fallback=None)
        max_freq = config.getfloat(section, "max-freq", fallback=None)
        l1_reg = config.getfloat(section, "l1-regulariser", fallback=0.)
        l2_reg = config.getfloat(section, "l2-regulariser", fallback=0.)
        dropout = config.getfloat(section, "dropout", fallback=0.)

        obj = cls(dt_max, df_max,
                  template_df, shared_layers, shift_layers,
                  freqs, f_low, f_high,
                  min_freq=min_freq, max_freq=max_freq,
                  l1_reg=l1_reg, l2_reg=l2_reg, dropout=dropout)
        return obj

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
            template_df = g['template_df'][0]
            layers = [int(l) for l in g['layers'][:]]
            min_freq = g['min_freq'][0]
            max_freq = g['max_freq'][0]
            l1_reg = g["l1_reg"][0]
            l2_reg = g["l2_reg"][0]
            dropout = g["dropout"][0]
            weights = []
            biases = []
            means = []
            variances = []
            for i in range(len(layers) + 1):
                weights += [g['weight_{0}'.format(i)][:]]
                biases += [g['bias_{0}'.format(i)][:]]

        obj = cls(dt_max, df_max,
                  template_df, layers, freqs, f_low, f_high,
                  min_freq=min_freq, max_freq=max_freq,
                  l1_reg=l1_reg, l2_reg=l2_reg, dropout=dropout)

        obj.weights = []
        obj.biases = []
        obj.trainable_weights = {}
        for i in range(len(layers) + 1):
            weight = tf.convert_to_tensor(weights[i], dtype=tf.float32)
            obj.weights += [tf.Variable(weight, trainable=True, dtype=tf.float32)]
            obj.trainable_weights['weight_{0}'.format(i)] = obj.weights[i]
            bias = tf.convert_to_tensor(biases[i], dtype=tf.float32)
            obj.biases += [tf.Variable(bias, trainable=True, dtype=tf.float32)]
            obj.trainable_weights['bias_{0}'.format(i)] = obj.biases[i]

        return obj


class Convolution1DTransform(BaseTransform):
    transformation_type = "convolution"

    def normalise(self, x):
        norm = tf.math.reduce_mean(tf.math.abs(x), axis=1, keepdims=True)
        denom = real_to_complex(norm) + 1e-7
        return x / denom

    def __init__(self, nkernel, kernel_df, max_df, max_dt, freqs, f_low, f_high,
                 l1_reg=0., l2_reg=0.):
        
        super().__init__(freqs, f_low, f_high, l1_reg=l1_reg, l2_reg=l2_reg)

        self.nkernel = nkernel
        self.kernel_df = kernel_df
        self.max_df = max_df
        self.max_dt = max_dt

        self.kernel_half_width = int(max_df // kernel_df)
        self.kernel_freqs = np.arange(self.kernel_half_width * 2 + 1) * self.kernel_df
        self.kernel_freqs -= self.kernel_freqs[self.kernel_half_width + 1]

        conv_freqs = np.arange(0., self.freqs[-1], self.kernel_df)

        self.to_conv = create_complex_interpolator(self.freqs.numpy(), conv_freqs)
        self.from_conv = create_complex_interpolator(conv_freqs, self.freqs.numpy(), temp_num=nkernel)

        kernels_real = tf.random.truncated_normal((nkernel, self.kernel_half_width * 2 + 1),
                                                  dtype=tf.float32)
        kernels_imag = tf.random.truncated_normal((nkernel, self.kernel_half_width * 2 + 1),
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
        
        return temp * shifter

    @tf.function
    def convolve(self, temp, training=False):

        kernels = self.kernels
        temp = self.to_conv(temp)

        kernels = tf.transpose(kernels, perm=[1, 0])
        kernels = tf.expand_dims(kernels, axis=0)
        kernels = tf.expand_dims(kernels, axis=2)

        temp = tf.transpose(temp, perm=[0, 2, 1])
        temp = tf.expand_dims(temp, axis=1)

        @tf.custom_gradient
        def _convolve(temp, kernels):
            kernels_real = tf.math.real(kernels)
            kernels_imag = tf.math.imag(kernels)

            temp_real = tf.math.real(temp)
            temp_imag = tf.math.imag(temp)

            padding = [[0, 0], [0, 0], [self.kernel_half_width, self.kernel_half_width], [0, 0]]

            ctemp_real = tf.nn.conv2d(temp_real, kernels_real, 1, padding, data_format='NHWC')
            ctemp_real -= tf.nn.conv2d(temp_imag, kernels_imag, 1, padding, data_format='NHWC')
            ctemp_imag = tf.nn.conv2d(temp_imag, kernels_real, 1, padding, data_format='NHWC')
            ctemp_imag += tf.nn.conv2d(temp_real, kernels_imag, 1, padding, data_format='NHWC')

            ctemp = tf.complex(ctemp_real, ctemp_imag)

            def grad(upstream):
                # upstream [batch, 1, width, nkernel]
                # change to [batch, width, 1, nkernel]
                upstream = tf.transpose(upstream, perm=[0, 2, 1, 3])
                up_real = tf.math.real(upstream)
                up_imag = tf.math.imag(upstream)

                # temp [batch, 1, width, 1]
                # change to [1, batch, width, 1]
                temp_k = tf.transpose(temp, perm=[1, 0, 2, 3])
                k_real = tf.math.real(temp_k)
                k_imag = tf.math.imag(temp_k)

                grad_real = tf.nn.conv2d(k_real, up_real, 1, padding, data_format='NHWC')
                grad_real += tf.nn.conv2d(k_imag, up_imag, 1, padding, data_format='NHWC')
                grad_imag = - tf.nn.conv2d(k_imag, up_real, 1, padding, data_format='NHWC')
                grad_imag += tf.nn.conv2d(k_real, up_imag, 1, padding, data_format='NHWC')

                grad_comp = tf.complex(grad_real, grad_imag)
                grad_comp = tf.transpose(grad_comp, perm=[0, 2, 1, 3])
                return None, grad_comp

            return ctemp, grad

        ctemp = _convolve(temp, kernels)
        ctemp = ctemp[:, 0, :, :]
        ctemp = tf.transpose(ctemp, perm=[0, 2, 1])

        div = tf.convert_to_tensor(self.kernel_half_width * 2 + 1, dtype=tf.complex64)
        ctemp = ctemp / div

        ctemp = self.from_conv(ctemp)
        return ctemp

    def transform(self, temp, sample, training=False):
        if tf.rank(temp) == 2:
            temp = tf.expand_dims(temp, 1)
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
        l1_reg = config.getfloat(section, "l1-regulariser", fallback=0.)
        l2_reg = config.getfloat(section, "l2-regulariser", fallback=0.)

        obj = cls(nkernel, kernel_df, max_df, max_dt, freqs, f_low, f_high,
                  l1_reg=l1_reg, l2_reg=l2_reg)
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
            g.attrs['transformation_type'] = self.transformation_type
            _ = g.create_dataset('nkernel', data=np.array([self.nkernel]))
            _ = g.create_dataset('kernel_df', data=np.array([self.kernel_df]))
            _ = g.create_dataset('max_df', data=np.array([self.max_df]))
            _ = g.create_dataset('max_dt', data=np.array([self.max_dt]))
            _ = g.create_dataset('kernels_real', data=self.kernels.numpy().real)
            _ = g.create_dataset('kernels_imag', data=self.kernels.numpy().imag)
            _ = g.create_dataset('dt_weights', data=self.dt_weights.numpy())
            _ = g.create_dataset("l1_reg", data=np.array([self.l1_reg]))
            _ = g.create_dataset("l2_reg", data=np.array([self.l2_reg]))

    @classmethod
    def from_file(cls, file_path, freqs, f_low, f_high, group=None):
        with h5py.File(file_path, 'r') as f:
            if group:
                g = f[group]
            else:
                g = f
            nkernel = g['nkernel'][0]
            kernel_df = g['kernel_df'][0]
            max_df = g['max_df'][0]
            max_dt = g['max_dt'][0]
            kernels_real = g['kernels_real'][:]
            kernels_imag = g['kernels_imag'][:]
            dt_weights = g['dt_weights'][:]
            l1_reg = g["l1_reg"][0]
            l2_reg = g["l2_reg"][0]

        obj = cls(nkernel, kernel_df, max_df, max_dt, freqs, f_low, f_high,
                  l1_reg=l1_reg, l2_reg=l2_reg)

        kernels_real = tf.convert_to_tensor(kernels_real, dtype=tf.float32)
        kernels_imag = tf.convert_to_tensor(kernels_imag, dtype=tf.float32)
        kernels = tf.complex(kernels_real, kernels_imag)
        obj.kernels = tf.Variable(kernels,
                                  dtype=tf.complex64, trainable=True,
                                  constraint=obj.normalise)

        obj.dt_weights = tf.Variable(dt_weights, dtype=tf.float32, trainable=True)

        obj.trainable_weights = {'kernels': obj.kernels,
                                 'dt_weights': obj.dt_weights}
        return obj

    def plot_model(self, bank, title=None):
        fig, ax = plt.subplots(nrows=self.nkernel,
                               figsize=(8, 6 * self.nkernel))

        kernels_real = tf.math.real(self.kernels)
        kernels_imag = tf.math.imag(self.kernels)

        dt = tf.math.tanh(self.dt_weights) * self.max_dt

        for i in range(self.nkernel):
            ax[i].plot(self.kernel_freqs, kernels_real.numpy()[i, :], label='real', alpha=0.75)
            ax[i].plot(self.kernel_freqs, kernels_imag.numpy()[i, :], label='imag', alpha=0.75)
            
            max_amp = max(np.max(np.abs(kernels_real.numpy()[i, :])),
                          np.max(np.abs(kernels_imag.numpy()[i, :])))
            ax[i].set_ylim([-max_amp, max_amp])

            ax[i].legend()
            ax[i].set_title('Time Shift (s) = {0}'.format(dt[i]), fontsize='small')
            ax[i].set_xlabel('Frequency (Hz)', fontsize='small')
            ax[i].set_ylabel('Amplitude', fontsize='small')
            ax[i].grid()

        if title:
            fig.suptitle(title, fontsize="large")

        return fig, ax


class CNNTransform(BaseTransform):
    transformation_type = "cnn"

    def __init__(self, kernel_width, kernel_out, kernel_df, freqs, f_low, f_high,
                 l1_reg=0., l2_reg=0.):
        
        super().__init__(freqs, f_low, f_high, l1_reg=l1_reg, l2_reg=l2_reg)

        if len(kernel_width) != len(kernel_out):
            raise ValueError("kernel_wdth and kernel_out must be the same length")
        self.kernel_width = kernel_width
        self.kernel_out = kernel_out
        self.kernel_df = kernel_df

        self.kernels = []
        self.trainable_weights = {}
        channels = 3
        for i, (w, out) in enumerate(zip(kernel_width, kernel_out)):
            kernel = tf.random.truncated_normal(
                (1, w, channels, out),
                dtype=tf.float32
            )
            kernel = tf.Variable(kernel, trainable=True)
            self.kernels += [kernel]
            self.trainable_weights['kernel_{0}'.format(i)] = kernel
            channels = out

        conv_freqs = np.arange(0., self.freqs[-1], self.kernel_df)
        self.to_conv = create_complex_interpolator(self.freqs.numpy(), conv_freqs)
        self.from_conv = create_complex_interpolator(conv_freqs, self.freqs.numpy(), temp_num=channels)

    #@tf.function
    def convolve(self, temp, training=False):

        total = real_to_complex(tf.reduce_sum(tf.math.abs(temp), axis=2, keepdims=True))
        kernels = self.kernels
        temp = self.to_conv(temp)

        temp = tf.transpose(temp, perm=[0, 2, 1])
        temp = tf.expand_dims(temp, axis=1)

        temp = [temp, tf.math.conj(temp), real_to_complex(tf.math.abs(temp))]
        temp = tf.concat(temp, axis=3)

        temp_real = tf.math.real(temp)
        temp_imag = tf.math.imag(temp)
        
        for k in self.kernels[:-1]:
            temp_real = tf.nn.conv2d(temp_real, k, 1, 'SAME', data_format='NHWC')
            temp_imag = tf.nn.conv2d(temp_imag, k, 1, 'SAME', data_format='NHWC')
            
            temp_real = tf.nn.relu(temp_real)
            temp_imag = tf.nn.relu(temp_imag)

        temp_real = tf.nn.conv2d(temp_real, self.kernels[-1], 1, 'SAME', data_format='NHWC')
        temp_imag = tf.nn.conv2d(temp_imag, self.kernels[-1], 1, 'SAME', data_format='NHWC')

        ctemp = tf.complex(temp_real, temp_imag)
        ctemp = ctemp[:, 0, :, :]
        ctemp = tf.transpose(ctemp, perm=[0, 2, 1])

        ctemp = self.from_conv(ctemp)
        denom = real_to_complex(tf.reduce_sum(tf.math.abs(ctemp), axis=2, keepdims=True))
        ctemp = ctemp / denom * total
        return ctemp

    def transform(self, temp, sample, training=False):
        if tf.rank(temp) == 2:
            temp = tf.expand_dims(temp, 1)
        temp = self.convolve(temp, training=training)
        return temp

    @classmethod
    def from_config(cls, config_file, freqs, f_low, f_high, section="model"):
        config = configparser.ConfigParser()
        config.read(config_file)

        kernel_width = [int(l) for l in config.get(section, "kernel-width").split(',')]
        kernel_out = [int(l) for l in config.get(section, "kernel-out").split(',')]
        kernel_df = config.getfloat(section, 'delta-f')
        l1_reg = config.getfloat(section, "l1-regulariser", fallback=0.)
        l2_reg = config.getfloat(section, "l2-regulariser", fallback=0.)

        obj = cls(kernel_width, kernel_out, kernel_df, freqs, f_low, f_high,
                  l1_reg=l1_reg, l2_reg=l2_reg)
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
            g.attrs['transformation_type'] = self.transformation_type
            _ = g.create_dataset('kernel_width', data=np.array(self.kernel_width))
            _ = g.create_dataset('kernel_out', data=np.array(self.kernel_out))
            _ = g.create_dataset('kernel_df', data=np.array([self.kernel_df]))
            _ = g.create_dataset("l1_reg", data=np.array([self.l1_reg]))
            _ = g.create_dataset("l2_reg", data=np.array([self.l2_reg]))
            for i in range(len(self.kernels)):
                _ = g.create_dataset("kernel_{0}".format(i),
                                     data=self.kernels[i].numpy())

    @classmethod
    def from_file(cls, file_path, freqs, f_low, f_high, group=None):
        with h5py.File(file_path, 'r') as f:
            if group:
                g = f[group]
            else:
                g = f
            kernel_width = list(g['kernel_width'][:])
            kernel_out = list(g['kernel_out'][:])
            kernel_df = g['kernel_df'][0]
            l1_reg = g["l1_reg"][0]
            l2_reg = g["l2_reg"][0]
            kernels = []
            for i in range(len(kernel_width)):
                kernel = g['kernel_{0}'.format(i)][:]
                kernels.append(kernel)

        obj = cls(kernel_width, kernel_out, kernel_df, freqs, f_low, f_high,
                  l1_reg=l1_reg, l2_reg=l2_reg)

        obj.kernels = []
        obj.trainable_weights = {}
        for i, k in enumerate(kernels):
            kernel = tf.convert_to_tensor(k, dtype=tf.float32)
            kernel = tf.Variable(kernel, trainable=True)
            obj.kernels += [kernel]
            obj.trainable_weights['kernel_{0}'.format(i)] = kernel

        return obj

    def plot_model(self, bank, title=None):
        fig, ax = plt.subplots(nrows=self.kernel_out[-1] + 1, ncols=3,
                               figsize=(48, 6 * (self.kernel_out[-1] + 1)))

        idxs = np.argsort(bank.table.mtotal)
        idxs = [idxs[0], idxs[len(idxs) // 2], idxs[-1]]

        for i, idx in enumerate(idxs):
            base_ftemp = bank[idx]
            ftemps = self.transform(base_ftemp.numpy()[np.newaxis, :].astype(np.complex64), None)
            
            base_temp = base_ftemp.to_timeseries()
            base_temp = base_temp.cyclic_time_shift(2.5)
            base_temp = base_temp.time_slice(base_temp.start_time, base_temp.start_time + 5.)

            ax[0, i].plot(base_temp.sample_times, base_temp.numpy(), alpha=0.75)
            ax[0, i].grid()

            for j in range(self.kernel_out[-1]):
                ftemp = FrequencySeries(ftemps[0, j, :].numpy().astype(np.complex128),
                                        self.delta_f, epoch=0.)
                temp = ftemp.to_timeseries()
                temp = temp.cyclic_time_shift(2.5)
                temp = temp.time_slice(temp.start_time, temp.start_time + 5.)

                ax[j+1, i].plot(temp.sample_times, temp.numpy(), alpha=0.75)
                ax[j+1, i].grid()

        if title:
            fig.suptitle(title, fontsize="large")

        return fig, ax


class DenseTransform(BaseTransform):
    transformation_type = "dense"

    def abs_constraint(self, x):
        return tf.math.maximum(x, tf.zeros_like(x))

    def phase_constraint(self, x):
        return tf.math.floormod(x, tf.ones_like(x) * 2. * np.pi)

    def __init__(self, template_df, layers, freqs, f_low, f_high,
                 min_freq=None, max_freq=None,
                 l1_reg=0., l2_reg=0.):
        
        super().__init__(freqs, f_low, f_high, l1_reg=l1_reg, l2_reg=l2_reg)

        self.template_df = template_df
        self.layers = layers

        if min_freq is None:
            min_freq = f_low
        if max_freq is None:
            max_freq = f_high

        self.min_freq = min_freq
        self.max_freq = max_freq

        template_freqs = np.arange(min_freq, max_freq, template_df)
        self.to_dense = create_complex_interpolator(self.freqs.numpy(), template_freqs)
        self.from_dense = create_complex_interpolator(template_freqs, self.freqs.numpy())

        self.weights_abs = []
        self.weights_phase = []
        self.biases = []
        self.trainable_weights = {}
        in_layer = 2 * len(template_freqs)
        for i, n in enumerate(layers + [len(template_freqs)]):
            weight_abs = tf.random.uniform(
                (in_layer, n),
                minval=0., maxval=tf.math.sqrt(6. / (in_layer + n)),
                dtype=tf.float32
            )
            weight_phase = tf.random.uniform(
                (in_layer, n),
                minval=0., maxval=2. * np.pi,
                dtype=tf.float32
            )
            self.weights_abs += [tf.Variable(weight_abs, trainable=True, dtype=tf.float32,
                                             constraint=self.abs_constraint)]
            self.weights_phase += [tf.Variable(weight_phase, trainable=True, dtype=tf.float32,
                                               constraint=self.phase_constraint)]
            self.trainable_weights['weight_{0}_abs'.format(i)] = self.weights_abs[i]
            self.trainable_weights['weight_{0}_phase'.format(i)] = self.weights_phase[i]
            if i < len(layers):
                bias = tf.zeros((n,), dtype=tf.float32)
                self.biases += [tf.Variable(bias, trainable=True, dtype=tf.float32)]
                self.trainable_weights['bias_{0}'.format(i)] = self.biases[i]
            in_layer = n

    def get_regulariser_loss(self):
        losses = []
        for n, w in self.trainable_weights.items():
            if n.endswith('phase'):
                continue
            l1_loss = 0.
            l2_loss = 0.
            if self.l1_reg:
                l1_loss = self.l1_reg * tf.reduce_sum(tf.math.abs(w))
            if self.l2_reg:
                l2_loss = self.l2_reg * tf.reduce_sum(tf.math.square(w))
            losses.append(l1_loss + l2_loss)
        loss = tf.math.add_n(losses)
        return loss

    def dense(self, temp, training=False):

        ntemp = temp / tf.reduce_mean(real_to_complex(tf.math.abs(temp)),
                                      axis=2, keepdims=True)
        ntemp = self.to_dense(ntemp)

        ntemp = tf.concat([ntemp, tf.math.conj(ntemp)], axis=2)
        real_temp = tf.math.real(ntemp)
        imag_temp = tf.math.imag(ntemp)

        for weight, phase, bias in zip(self.weights_abs[:-1], self.weights_phase[:-1], self.biases):
            real_w = weight * tf.math.cos(phase)
            imag_w = weight * tf.math.sin(phase)
            
            real_keep = real_temp
            imag_keep = imag_temp

            real_temp = tf.matmul(real_keep, real_w) - tf.matmul(imag_keep, imag_w)
            imag_temp = tf.matmul(real_keep, imag_w) + tf.matmul(imag_keep, real_w)

            mod = (real_temp ** 2. + imag_temp ** 2.) ** 0.5
            diff = tf.maximum(1 + bias / mod, tf.zeros_like(mod))
            if training:
                diff = tf.nn.dropout(diff, 0.5)

            real_temp = diff * real_temp
            imag_temp = diff * imag_temp

        real_w = self.weights_abs[-1] * tf.math.cos(self.weights_phase[-1])
        imag_w = self.weights_abs[-1] * tf.math.sin(self.weights_phase[-1])
            
        real_keep = real_temp
        imag_keep = imag_temp

        real_temp = tf.matmul(real_keep, real_w) - tf.matmul(imag_keep, imag_w)
        imag_temp = tf.matmul(real_keep, imag_w) + tf.matmul(imag_keep, real_w)

        ntemp = tf.complex(real_temp, imag_temp)

        ntemp = self.from_dense(ntemp)
        ntemp = ntemp / tf.reduce_sum(real_to_complex(tf.math.abs(ntemp)),
                                      axis=2, keepdims=True)

        ntemp = ntemp * real_to_complex(tf.math.abs(temp))
        return ntemp

    def transform(self, temp, sample, training=False):
        if tf.rank(temp) == 2:
            temp = tf.expand_dims(temp, 1)
        temp = self.dense(temp, training=training)
        return temp

    @classmethod
    def from_config(cls, config_file, freqs, f_low, f_high, section="model"):
        config = configparser.ConfigParser()
        config.read(config_file)

        template_df = config.getfloat(section, "template-df")
        layers = [int(l) for l in config.get(section, "layers").split(',')]
        min_freq = config.getfloat(section, "min-freq", fallback=None)
        max_freq = config.getfloat(section, "max-freq", fallback=None)
        l1_reg = config.getfloat(section, "l1-regulariser", fallback=0.)
        l2_reg = config.getfloat(section, "l2-regulariser", fallback=0.)

        obj = cls(template_df, layers, freqs, f_low, f_high,
                  min_freq=min_freq, max_freq=max_freq,
                  l1_reg=l1_reg, l2_reg=l2_reg)
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
            g.attrs['transformation_type'] = self.transformation_type
            _ = g.create_dataset('template_df', data=np.array([self.template_df]))
            _ = g.create_dataset('layers', data=np.array(self.layers))
            _ = g.create_dataset('min_freq', data=np.array([self.min_freq]))
            _ = g.create_dataset('max_freq', data=np.array([self.max_freq]))
            _ = g.create_dataset("l1_reg", data=np.array([self.l1_reg]))
            _ = g.create_dataset("l2_reg", data=np.array([self.l2_reg]))
            for i in range(len(self.weights_abs)):
                _ = g.create_dataset("weight_{0}_abs".format(i),
                                     data=self.weights_abs[i].numpy())
                _ = g.create_dataset("weight_{0}_phase".format(i),
                                     data=self.weights_phase[i].numpy())
                if i != (len(self.weights_abs) - 1):
                    _ = g.create_dataset("bias_{0}".format(i),
                                         data=self.biases[i].numpy())

    @classmethod
    def from_file(cls, file_path, freqs, f_low, f_high, group=None):
        with h5py.File(file_path, 'r') as f:
            if group:
                g = f[group]
            else:
                g = f
            template_df = g['template_df'][0]
            layers = [int(l) for l in g['layers'][:]]
            min_freq = g['min_freq'][0]
            max_freq = g['max_freq'][0]
            l1_reg = g["l1_reg"][0]
            l2_reg = g["l2_reg"][0]
            weights_abs = []
            weights_phase = []
            biases = []
            for i in range(len(layers) + 1):
                weights_abs += [g['weight_{0}_abs'.format(i)][:]]
                weights_phase += [g['weight_{0}_phase'.format(i)][:]]
                if i < len(layers):
                    biases += [g['bias_{0}'.format(i)][:]]

        obj = cls(template_df, layers, freqs, f_low, f_high,
                  min_freq=min_freq, max_freq=max_freq,
                  l1_reg=l1_reg, l2_reg=l2_reg)

        obj.weights_abs = []
        obj.weights_phase = []
        obj.biases = []
        obj.trainable_weights = {}
        for i in range(len(layers) + 1):
            weight_abs = tf.convert_to_tensor(weights_abs[i], dtype=tf.float32)
            weight_phase = tf.convert_to_tensor(weights_phase[i], dtype=tf.float32)
            obj.weights_abs += [tf.Variable(weight_abs, trainable=True, dtype=tf.float32,
                                            constraint=obj.abs_constraint)]
            obj.weights_phase += [tf.Variable(weight_phase, trainable=True, dtype=tf.float32,
                                              constraint=obj.phase_constraint)]
            obj.trainable_weights['weight_{0}_abs'.format(i)] = obj.weights_abs[i]
            obj.trainable_weights['weight_{0}_phase'.format(i)] = obj.weights_phase[i]
            if i < len(layers):
                bias = tf.convert_to_tensor(biases[i], dtype=tf.float32)
                obj.biases += [tf.Variable(bias, trainable=True, dtype=tf.float32)]
                obj.trainable_weights['bias_{0}'.format(i)] = obj.biases[i]

        return obj

    def plot_model(self, bank, title=None):
        fig, ax = plt.subplots(nrows=2, ncols=3,
                               figsize=(48, 12))

        idxs = np.argsort(bank.table.mtotal)
        idxs = [idxs[0], idxs[len(idxs) // 2], idxs[-1]]

        for i, idx in enumerate(idxs):
            base_ftemp = bank[idx]
            ftemp = self.transform(base_ftemp.numpy()[np.newaxis, :].astype(np.complex64), None)
            
            base_temp = base_ftemp.to_timeseries()
            base_temp = base_temp.cyclic_time_shift(2.5)
            base_temp = base_temp.time_slice(base_temp.start_time, base_temp.start_time + 5.)

            ax[0, i].plot(base_temp.sample_times, base_temp.numpy(), alpha=0.75)
            ax[0, i].grid()

            ftemp = FrequencySeries(ftemp[0, 0, :].numpy().astype(np.complex64),
                                    self.delta_f, epoch=0.)
            temp = ftemp.to_timeseries()
            temp = temp.cyclic_time_shift(2.5)
            temp = temp.time_slice(temp.start_time, temp.start_time + 5.)

            ax[1, i].plot(temp.sample_times, temp.numpy(), alpha=0.75)
            ax[1, i].grid()
            
        if title:
            fig.suptitle(title, fontsize="large")

        return fig, ax


def select_transformation(transformation_key):
    options = {
        'polyshift': PolyShiftTransform,
        'netshift': NetShiftTransform,
        'quadnetshift': QuadNetShiftTransform,
        'convolution': Convolution1DTransform,
        'cnn': CNNTransform,
        'dense': DenseTransform
    }

    if transformation_key in options.keys():
        return options[transformation_key]

    raise ValueError("{0} is not a valid transformation, select from {1}".format(transformation_key, options.keys()))


def load_transformation(file_path, freqs, f_low, f_high, group=None):
    with h5py.File(file_path, 'r') as f:
        if group:
            g = f[group]
        else:
            g = f
        transformation_type = g.attrs["transformation_type"]
    transformation_class = select_transformation(transformation_type)
    return transformation_class.from_file(file_path, freqs, f_low, f_high, group=group)
