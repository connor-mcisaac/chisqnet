import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np


def real_to_complex(real):
    imag = tf.zeros_like(real, dtype=tf.float64)
    return tf.complex(real, imag)


class MatchedFilter(object):

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

    def add_batch_axis(self, x, batch):
        x = tf.expand_dims(x, axis=0)
        x = tf.repeat(x, batch, axis=0)
        return x

    def add_temp_axis(self, x, ntemp):
        x = tf.expand_dims(x, axis=1)
        x = tf.repeat(x, ntemp, axis=1)
        return x

    def add_length_axis(self, x, full=False):
        if full:
            length = self.full_length
        else:
            length = self.length
        x = tf.expand_dims(x, axis=-1)
        x = tf.repeat(x, length, axis=-1)
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
        left_pad = tf.zeros(list(shape[:-1]) + [self.kmin], dtype=tf.complex128)
        if length == self.kmax:
            x = tf.concat([left_pad, x], axis=-1)
        else:
            right_pad = tf.zeros(list(shape[:-1]) + [length - self.kmax], dtype=tf.complex128)
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

    def sigma(self, a, b, psd):
        inner = self.inner(a, b, psd)
        return self.inner_to_sigma(inner)

    def __call__(self, data, psd, temp):

        if tf.size(tf.shape(temp)) == 3:
            ntemp = tf.shape(temp)[1]
            data = self.add_temp_axis(data, ntemp)
            psd = self.add_temp_axis(psd, ntemp)

        data = self.cut(data)
        psd = self.cut(psd)
        temp = self.cut(temp)

        sigma = self.sigma(temp, temp, psd)
        snr_tilde = self.inner(temp, data, psd) / sigma

        snr_tilde = self.pad(snr_tilde, full=True)
        snr = tf.signal.ifft(snr_tilde) * self.full_length
        return snr

    def get_ortho(self, temp, base, psd):
        
        ntemp = tf.shape(temp)[1]

        temp = self.cut(temp)
        base = self.cut(base)
        psd = self.cut(psd)

        base = self.add_temp_axis(base, ntemp)
        psd = self.add_temp_axis(psd, ntemp)

        temp_sigma = self.sigma(temp, temp, psd)
        base_sigma = self.sigma(base, base, psd)

        temp /= temp_sigma
        base /= base_sigma

        overlap = self.inner(base, temp, psd)
        overlap = tf.math.reduce_sum(overlap, axis=-1, keepdims=True)
        overlap = tf.repeat(overlap, self.cut_length, axis=-1)

        ortho = (temp - overlap * base) / (1 - overlap * tf.math.conj(overlap)) ** 0.5
        ortho = self.pad(ortho)

        return ortho

    def shift_dt(self, temp, dt):

        batch = tf.shape(temp)[0]
        ntemp = tf.shape(dt)[1]

        freqs = self.add_batch_axis(self.freqs, batch)
        freqs = self.add_temp_axis(freqs, ntemp)

        dt = self.add_length_axis(dt)

        freqs = real_to_complex(freqs)
        dt = real_to_complex(dt)

        shifter = tf.math.exp(tf.constant(1j * np.pi) * freqs * dt)

        if tf.size(tf.shape(temp)) == 2:
            temp = self.add_temp_axis(temp, ntemp)
        
        return temp * shifter

    def shift_df(self, temp, df):

        batch = tf.shape(temp)[0]
        ntemp = tf.shape(df)[1]

        temp_tilde = self.cut(temp)
        temp_tilde = self.pad(temp_tilde, full=True)
        temp = tf.signal.ifft(temp_tilde)

        times = self.add_batch_axis(self.times, batch)
        times = self.add_temp_axis(times, ntemp)

        df = self.add_length_axis(df, full=True)

        times = real_to_complex(times)
        df = real_to_complex(df)

        shifter = tf.math.exp(tf.constant(1j * np.pi) * df * times)

        if tf.size(tf.shape(temp)) == 2:
            temp = self.add_temp_axis(temp, ntemp)

        temp = temp * shifter
        temp_tilde = tf.signal.fft(temp)

        return temp_tilde[..., :self.length]

    def shift_dt_df(self, temp, dt, df):
        temp = self.shift_dt(temp, dt)
        temp = self.shift_df(temp, df)
        return temp
