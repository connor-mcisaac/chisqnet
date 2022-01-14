import tensorflow as tf
import tensorflow_probability as tfp


class Loss(object):

    @classmethod
    def from_cli(cls, args):
        kwarg_strings = args.loss_keyword_args
        kwargs = {}
        for kwarg_str in kwarg_strings:
            k, v = kwarg_str.split(':')
            kwargs[k] = float(v)
        return cls(**kwargs)


class AbsoluteLoss(Loss):

    def __init__(self, threshold=4.):
        self.threshold = threshold

    def __call__(self, snr_prime, snr, injs, training=False):
        snr_prime_thresh = tf.maximum(snr_prime, tf.ones_like(snr_prime) * self.threshold)

        trig_loss = snr_prime_thresh - self.threshold
        inj_loss = snr - snr_prime

        batch_loss = tf.where(injs, inj_loss, trig_loss)
        return batch_loss


class SquaredLoss(Loss):

    def __init__(self, threshold=4.):
        self.threshold = threshold

    def __call__(self, snr_prime, snr, injs, training=False):
        snr_prime_thresh = tf.maximum(snr_prime, tf.ones_like(snr_prime) * self.threshold)
        
        trig_loss = (snr_prime_thresh - self.threshold) ** 2.
        inj_loss = (snr - snr_prime) ** 2.
        
        batch_loss = tf.where(injs, inj_loss, trig_loss)
        return batch_loss


class StatLoss(Loss):

    def __init__(self, snr_cut=4., threshold=6., alpha_below=6., alpha_above=4.):
        self.snr_cut = snr_cut
        self.threshold = threshold
        self.alpha_below = alpha_below
        self.alpha_above = alpha_above
        self.const = 1. / alpha_above - 1. / alpha_below + tf.math.exp(- alpha_below * (snr_cut - threshold)) / alpha_below

    def __call__(self, snr_prime, snr, injs, training=False):
        snr_prime_thresh = tf.maximum(snr_prime, tf.ones_like(snr_prime) * self.snr_cut)

        loss_above = - self.alpha_above * (snr_prime_thresh - self.threshold) - tf.math.log(self.const)
        loss_below = - self.alpha_below * (snr_prime_thresh - self.threshold) - tf.math.log(self.const)

        above_thresh = tf.greater(snr_prime, tf.ones_like(snr_prime) * self.threshold)
        sign = 2. * tf.cast(injs, tf.float32) - 1.

        loss = tf.where(above_thresh, loss_above, loss_below)
        return sign * loss


class ProbLoss(StatLoss):

    def __init__(self, snr_cut=4., threshold=6., alpha_below=6., alpha_above=4., epsilon=1e-6):
        super().__init__(snr_cut=4., threshold=6., alpha_below=6., alpha_above=4.)
        self.epsilon = tf.constant(epsilon, tf.float32)

    def __call__(self, snr_prime, snr, injs, training=False):
        snr_prime_thresh = tf.maximum(snr_prime, tf.ones_like(snr_prime) * self.snr_cut)
        label = 1. - tf.cast(injs, tf.float32)

        log_prob_above = - self.alpha_above * (snr_prime_thresh - self.threshold) - tf.math.log(self.alpha_above)
        log_prob_below = - self.alpha_below * (snr_prime_thresh - self.threshold) - tf.math.log(self.alpha_below)
        log_prob_below = tfp.math.log_add_exp(log_prob_below,  tf.math.log(1. / self.alpha_above - 1. / self.alpha_below))

        log_prob_above = log_prob_above - tf.math.log(self.const)
        log_prob_below = log_prob_below - tf.math.log(self.const)

        above_thresh = tf.greater(snr_prime, tf.ones_like(snr_prime) * self.threshold)

        log_prob = tf.where(above_thresh, log_prob_above, log_prob_below)
        log_prob = tf.minimum(log_prob, 0. - self.epsilon)

        loss = - (label * log_prob + (1. - label) * tfp.math.log_sub_exp(0., log_prob))
        return loss


def select_loss(loss_key):
    options = {
        'absolute': AbsoluteLoss,
        'squared': SquaredLoss,
        'stat': StatLoss,
        'prob': ProbLoss
    }

    if loss_key in options.keys():
        return options[loss_key]

    raise ValueError("{0} is not a valid loss, select from {1}".format(loss_key, options.keys()))
