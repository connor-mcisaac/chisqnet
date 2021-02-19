import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def real_to_complex(real):
    imag = tf.zeros_like(real, dtype=tf.float64)
    return tf.complex(real, imag)


class BaseFilter(object):

    def __init__(self, freqs, f_low, f_high):

        self.freqs = tf.convert_to_tensor(freqs, tf.float64)
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

        self.times = tf.range(self.full_length, delta=1., dtype=tf.float64) * self.delta_t

    def add_axis(self, x, axis, length):
        x = tf.expand_dims(x, axis=axis)
        x = tf.repeat(x, length, axis=axis)
        return x

    def cut(self, x):
        if tf.shape(x)[-1] != self.length:
            raise ValueError("This function should only be used on a tensor"
                             + " with last dim length " + str(self.length)
                             + ", x has length " + str(tf.shape(x)[-1]))
        return x[..., self.kmin:self.kmax]

    def pad(self, x, full=False):
        if full:
            length = self.full_length
        else:
            length = self.length
        shape = tf.shape(x)
        if shape[-1] != self.cut_length:
            raise ValueError("This function should only be used on a tensor"
                             + " with last dim length " + str(self.cut_length)
                             + ", x has length " + str(tf.shape(x)[-1]))
        left_shape = tf.concat([shape[:-1], tf.constant([self.kmin])], axis=-1)
        left_pad = tf.zeros(left_shape, dtype=tf.complex128)
        if length == self.kmax:
            x = tf.concat([left_pad, x], axis=-1)
        else:
            right_shape = tf.concat([shape[:-1], tf.constant([length - self.kmax])], axis=-1)
            right_pad = tf.zeros(right_shape, dtype=tf.complex128)
            x = tf.concat([left_pad, x, right_pad], axis=-1)        
        return x

    def inner(self, a, b, psd):
        a_star = tf.math.conj(a)
        norm = 4. * self.delta_f
        inner = a_star * b * norm / psd
        return inner

    def inner_to_sigma(self, inner):
        cplx = tf.math.reduce_sum(inner, axis=-1, keepdims=True)
        sigma = real_to_complex(tf.math.real(cplx) ** 0.5)
        return tf.repeat(sigma, tf.shape(inner)[-1], axis=-1)

    def sigma(self, x, psd):
        inner = self.inner(x, x, psd)
        return self.inner_to_sigma(inner)
    

class MatchedFilter(BaseFilter):

    def __call__(self, data, psd, temp):

        if tf.rank(temp) == 3:
            ntemp = tf.shape(temp)[1]
            data = self.add_axis(data, 1, ntemp)
            psd = self.add_axis(psd, 1, ntemp)

        data = self.cut(data)
        psd = self.cut(psd)
        temp = self.cut(temp)

        sigma = self.sigma(temp, psd)
        snr_tilde = self.inner(temp, data, psd) / sigma

        snr_tilde = self.pad(snr_tilde, full=True)
        snr = tf.signal.ifft(snr_tilde) * self.full_length
        return snr

class BaseTransform(BaseFilter):

    def get_ortho(self, temp, base, psd):
        
        ntemp = tf.shape(temp)[1]

        temp = self.cut(temp)
        base = self.cut(base)
        psd = self.cut(psd)

        base = self.add_axis(base, 1, ntemp)
        psd = self.add_axis(psd, 1, ntemp)

        temp_sigma = self.sigma(temp, psd)
        base_sigma = self.sigma(base, psd)

        temp_norm = temp / temp_sigma
        base_norm = base / base_sigma

        inner = self.inner(base_norm, temp_norm, psd)
        overlap = tf.math.reduce_sum(inner, axis=-1, keepdims=True)
        overlap = tf.repeat(overlap, self.cut_length, axis=-1)

        ortho = ((temp_norm - overlap * base_norm)
                 / (1 - overlap * tf.math.conj(overlap)) ** 0.5)
        ortho = self.pad(ortho)

        return ortho


class ShiftTransform(BaseTransform):

    def __init__(self, nshifts, degree, dt, df, param_max,
                 freqs, f_low, f_high):

        super().__init__(freqs, f_low, f_high)

        if degree < 0:
            raise ValueError("degree must be >= 0")
        self.nshifts = nshifts
        self.degree = degree

        offset = np.pi / nshifts
        dt_base = 1. * dt * np.sin(2. * np.pi * np.arange(nshifts) / nshifts + offset)
        df_base = 1. * df * np.cos(2. * np.pi * np.arange(nshifts) / nshifts + offset)

        def create_constraint(max_value):
            def constraint(x):
                x = tf.clip_by_value(x, - 1. * max_value, max_value)
                return x
            return constraint

        dts = [dt_base * 1. / (param_max ** i) / (degree + 1) for i in range(degree + 1)]
        dts = np.stack(dts, axis=-1)
        dt_lims = [dt * 1. / (param_max ** i) for i in range(degree + 1)]
        dt_lims = np.stack(dt_lims, axis=-1)
        self.dts = tf.Variable(dts, trainable=True, dtype=tf.float64,
                               constraint=create_constraint(dt_lims))

        dfs = [df_base * 1. / (param_max ** i) / (degree + 1) for i in range(degree + 1)]
        dfs = np.stack(dfs, axis=-1)
        df_lims = [df * 1. / (param_max ** i) for i in range(degree + 1)]
        df_lims = np.stack(df_lims, axis=-1)
        self.dfs = tf.Variable(dfs, trainable=True, dtype=tf.float64,
                               constraint=create_constraint(df_lims))

    @tf.custom_gradient
    def mul_grad(self, x, y):
        z = x * y
        def grad(upstream):
            dz_dx = 1. / y
            return dz_dx * upstream, None
        return z, grad            

    def get_dt(self, param):
        batch = tf.shape(param)[0]
        dts = self.add_axis(self.dts, 0, batch)

        param = self.add_axis(param, 1, self.nshifts)
        param = self.add_axis(param, 2, self.degree + 1)
        
        power = tf.range(0, self.degree + 1, delta=1., dtype=tf.float64)
        power = self.add_axis(power, 0, batch)
        power = self.add_axis(power, 1, self.nshifts)
        
        params = tf.math.pow(param, power)

        terms = self.mul_grad(dts, params)
        dt = tf.math.reduce_sum(terms, axis=-1)

        return dt

    def get_df(self, param):
        batch = tf.shape(param)[0]
        dfs = self.add_axis(self.dfs, 0, batch)

        param = self.add_axis(param, 1, self.nshifts)
        param = self.add_axis(param, 2, self.degree + 1)
        
        power = tf.range(0, self.degree + 1, delta=1., dtype=tf.float64)
        power = self.add_axis(power, 0, batch)
        power = self.add_axis(power, 1, self.nshifts)
        
        params = tf.math.pow(param, power)

        terms = self.mul_grad(dfs, params)
        df = tf.math.reduce_sum(terms, axis=-1)

        return df

    def shift_dt(self, temp, param):

        dt = self.get_dt(param)
        dt = self.add_axis(dt, 2, self.length)

        batch = tf.shape(temp)[0]

        freqs = self.add_axis(self.freqs, 0, batch)
        freqs = self.add_axis(freqs, 1, self.nshifts)

        shifter = tf.complex(tf.math.cos(- 2. * np.pi * freqs * dt),
                             tf.math.sin(- 2. * np.pi * freqs * dt))

        if tf.rank(temp) == 2:
            temp = self.add_axis(temp, 1, self.nshifts)
        
        return temp * shifter

    def shift_df(self, temp, param):

        df = self.get_df(param)
        mod = tf.math.floormod(df, self.delta_f)
        df = df - tf.stop_gradient(mod)
        df = self.add_axis(df, 2, self.full_length)

        shape = tf.shape(temp)
        batch = shape[0]

        temp = tf.signal.irfft(temp * self.full_length)

        times = self.add_axis(self.times, 0, batch)
        times = self.add_axis(times, 1, self.nshifts)

        shifter = tf.complex(tf.math.cos(2. * np.pi * df * times),
                             tf.math.sin(2. * np.pi * df * times))

        if tf.rank(temp) == 2:
            temp = self.add_axis(temp, 1, self.nshifts)

        temp = real_to_complex(temp)
        temp = temp * shifter
        temp = tf.signal.fft(temp)

        temp = temp[..., :self.length]
        df = df[..., :self.length]

        freqs = self.add_axis(self.freqs, 0, batch)
        freqs = self.add_axis(freqs, 1, self.nshifts)

        low = freqs - self.delta_f / 2. - df
        low = low / tf.math.abs(low)
        low = tf.math.maximum(low, tf.zeros_like(low))
        low = real_to_complex(low)

        high = tf.math.reduce_max(freqs) - freqs - self.delta_f / 2. + df
        high = high / tf.math.abs(high)
        high = tf.math.maximum(high, tf.zeros_like(high))
        high = real_to_complex(high)

        return temp * low * high

    def shift_dt_df(self, temp, param):
        temp = self.shift_df(temp, param)
        temp = self.shift_dt(temp, param)
        return temp
