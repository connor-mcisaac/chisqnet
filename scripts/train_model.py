#!/home/connor.mcisaac/envs/tensor-imbh/bin/python3.6

import argparse
import logging, time
import h5py
import numpy as np
from pycbc import waveform
from pycbc.types import zeros, float32, complex64
from filtering import MatchedFilter, ShiftTransform, Convolution1DTransform
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


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

parser = argparse.ArgumentParser()

# Gather inputs from planning step, Injection file is included in strain options group
parser.add_argument("--sample-file", required=True,
                    help="Sample file to be used in training")

# Gather arguments for filtering/bank
parser.add_argument("--bank-file", required=True,
                    help="The bank file to be used to generate templates")
parser.add_argument("--low-frequency-cutoff", type=float,
                  help="The low frequency cutoff to use for filtering (Hz)")
parser.add_argument("--enable-bank-start-frequency", action='store_true',
                  help="Read the starting frequency of template waveforms"
                       " from the template bank.")
parser.add_argument("--max-template-length", type=float,
                  help="The maximum length of a template is seconds. The "
                       "starting frequency of the template is modified to "
                       "ensure the proper length")
waveform.bank.add_approximant_arg(parser)
parser.add_argument("--order", type=int,
                  help="The integer half-PN order at which to generate"
                       " the approximant. Default is -1 which indicates to use"
                       " approximant defined default.", default=-1,
                       choices = np.arange(-1, 9, 1))
taper_choices = ["start","end","startend"]
parser.add_argument("--taper-template", choices=taper_choices,
                    help="For time-domain approximants, taper the start and/or"
                    " end of the waveform before FFTing.")

# Gather arguments for training
parser.add_argument("--batch-size", required=True, type=int,
                    help="The number of samples to analyse in one batch")
parser.add_argument("--epochs", required=True, type=int,
                     help="The number of times to analyse the full dataset")
parser.add_argument("--shuffle", action="store_true",
                    help="If given, shuffle the order of analysed segments each epoch")
parser.add_argument("--snr-cut-width", default=0.1, type=float,
                    help="The width around each trigger to cut from the SNR timeseries")
parser.add_argument("--learning-rate", default=0.001, type=float,
                    help="The learning rate passed to the training optimizer")

# Gather arguments for shift transform
parser.add_argument("--shift-num", default=8, type=int,
                    help="The number of shift transformations to train")
parser.add_argument("--poly-degree", default=2, type=int,
                    help="The degree of the polynomial use in the shift transformations")
parser.add_argument("--base-time-shift", default=0.01, type=float,
                    help="The maximum time shift that can be caused by a single term in "
                         "the polynomial.")
parser.add_argument("--base-freq-shift", default=10., type=float,
                    help="The maximum frequency shift that can be caused by a single term in "
                         "the polynomial.")
parser.add_argument("--shift-param", default="mtotal", type=str,
                    help="The parameter used to calculate the time and frequency shifts")

# Gather additional options
parser.add_argument("--verbose", action='store_true')

args = parser.parse_args()

if args.verbose:
    log_level = logging.DEBUG
else:
    log_level = logging.WARNING
logging.basicConfig(format='%(asctime)s : %(message)s', level=log_level)

# Check option groups
samples = SampleFile(args.sample_file)

template_mem = zeros(samples.tlen, dtype=complex64)

if args.enable_bank_start_frequency:
    low_frequency_cutoff = None
else:
    low_frequency_cutoff = args.low_frequency_cutoff

bank = waveform.FilterBank(args.bank_file, samples.flen, samples.delta_f,
                           low_frequency_cutoff=low_frequency_cutoff,
                           dtype=complex64, phase_order=args.order,
                           taper=args.taper_template, approximant=args.approximant,
                           out=template_mem, max_template_length=args.max_template_length)

freqs = np.arange(samples.flen) * samples.delta_f

match = MatchedFilter(freqs, samples.flow, samples.sample_rate // 2)

min_param, max_param = samples.get_param_min_max(args.shift_param)

transform = ShiftTransform(args.shift_num, args.poly_degree,
                           args.base_time_shift,
                           args.base_freq_shift,
                           max_param,
                           freqs, samples.flow,
                           samples.sample_rate // 2)

params = tf.Variable(np.linspace(min_param, max_param, num=25, endpoint=True),
                     dtype=tf.float32)
dts = transform.get_dt(params).numpy()
dfs = transform.get_df(params).numpy()
params = params.numpy()

plt.figure(figsize=(8, 6))
for i in range(args.shift_num):
    plt.scatter(dts[:, i], dfs[:, i], c=params, cmap="cool")
plt.xlim([-0.02, 0.02])
plt.ylim([-15, 15])
plt.xlabel('Time Shift (s)', fontsize='large')
plt.ylabel('Frequency Shift (Hz)', fontsize='large')
plt.grid()
cbar = plt.colorbar()
cbar.ax.set_ylabel('Total Mass', fontsize='large')
plt.savefig('/home/connor.mcisaac/public_html/imbh/mass_shifts_e0.png', bbox='tight')
plt.close()

chi2 = tfp.distributions.Chi2(tf.constant(1., dtype=tf.float32))

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate)

for i in range(args.epochs):

    order = np.arange(samples.num)
    if args.shuffle:
        np.random.shuffle(order)

    nbatches = int(np.ceil(1. * samples.num / args.batch_size))

    for j in range(nbatches):

        lidx = int(1. * j * samples.num / nbatches)
        hidx = int(1. * (j + 1) * samples.num / nbatches)

        idxs = order[lidx:hidx]
        idxs = np.sort(idxs)

        logging.info("Starting batch")

        start = time.time()

        segs, psds, cuts, tids, injs = samples.get_samples(idxs)

        temp = np.stack([bank[tid].numpy().copy() for tid in tids])
        gather_idxs = np.zeros((len(idxs), cuts[0, 1] - cuts[0, 0], 2), dtype=int)
        for k, cut in enumerate(cuts):
            gather_idxs[k, :, 0] = k
            gather_idxs[k, :, 1] = np.arange(cut[0], cut[1])
        param = samples.get_params(args.shift_param, idxs)

        segs = tf.convert_to_tensor(segs, dtype=tf.complex64)
        psds = tf.convert_to_tensor(psds, dtype=tf.float32)
        temp = tf.convert_to_tensor(temp, dtype=tf.complex64)
        injs = tf.convert_to_tensor(injs, dtype=tf.bool)
        gather_idxs = tf.convert_to_tensor(gather_idxs, dtype=tf.int64)
    
        param = tf.convert_to_tensor(param, dtype=tf.float32)

        logging.info("Batch inputs read")

        with tf.GradientTape() as tape:

            chi_temps = transform.transform(temp, param)
            logging.info("Templates transformed")

            chi_orthos, ortho_lgc =  transform.get_ortho(chi_temps, temp, psds)
            logging.info("Orthogonal templates created")

            snr, _ = match.matched_filter(temp, segs, psds)
            snr = snr[:, 0, :]

            snr_cut = tf.gather_nd(snr, gather_idxs)
            max_snr = tf.math.argmax(snr_cut, axis=-1)

            idx = max_snr + gather_idxs[:, 0, 1]
            gather_max = tf.stack([tf.range(len(idxs), dtype=tf.int64), idx], axis=-1)

            snr_max = tf.gather_nd(snr, gather_max)

            chis, match_lgc = match.matched_filter(chi_orthos, segs, psds, idx=idx)

            lgc = tf.math.logical_and(ortho_lgc, match_lgc)
            mask = tf.cast(lgc, tf.float32)[:, :, 0]
            ortho_num = tf.reduce_sum(mask, axis=1)

            chis = chis * tf.stop_gradient(mask)

            logging.info("SNRs calculated")

            chisq = tf.math.reduce_sum(chis ** 2., axis=1) / 2. / tf.stop_gradient(ortho_num)

            chisq_thresh = tf.math.maximum(chisq, tf.ones_like(chisq, dtype=tf.float32))

            snr_prime = snr_max / chisq_thresh ** 0.5

            logging.info("SNR' calculated")

            snr_logp = - chi2.log_prob(snr_prime)
            chi_logp = - chi2.log_prob(chisq)

            num_inj = tf.math.reduce_sum(tf.cast(injs, tf.float32)).numpy()

            trig_weight = tf.ones_like(injs, dtype=tf.float32)
            trig_weight /= (len(idxs) - num_inj)

            inj_weight = tf.ones_like(injs, dtype=tf.float32)
            inj_num = tf.ones_like(injs, dtype=tf.float32) * num_inj
            inj_weight /= tf.maximum(inj_weight, inj_num)

            batch_loss = tf.where(injs,
                                  chi_logp * inj_weight,
                                  snr_logp * trig_weight)
            loss = tf.reduce_mean(batch_loss)

            logging.info("Loss calculated")

        gradients = tape.gradient(loss, transform.trainable_weights,
                                  unconnected_gradients='none')
        #gradients = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
        #             for grad in gradients]
        logging.info("Gradients calculated")

        optimizer.apply_gradients(zip(gradients, transform.trainable_weights))
        #for z in range(len(gradients)):
        #    transform.trainable_weights[z].assign_sub(args.learning_rate * gradients[z])

        logging.info("Gradients applied")
        logging.info("Completed batch")

        duration = time.time() - start
        print("Epoch: {0:5d}/{1:5d} Batch: {2:5d}/{3:5d}, ".format(i + 1, args.epochs, j + 1, nbatches)
              + "Loss: {0:12.6f}, Time: {1:8.2f}".format(loss, duration))

        params = tf.Variable(np.linspace(min_param, max_param, num=25, endpoint=True),
                             dtype=tf.float32)
        dts = transform.get_dt(params).numpy()
        dfs = transform.get_df(params).numpy()
        params = params.numpy()
    
    plt.figure(figsize=(8, 6))
    for k in range(args.shift_num):
        plt.scatter(dts[:, k], dfs[:, k], c=params, cmap="cool")
    plt.xlim([-0.02, 0.02])
    plt.ylim([-15, 15])
    plt.xlabel('Time Shift (s)', fontsize='large')
    plt.ylabel('Frequency Shift (Hz)', fontsize='large')
    plt.grid()
    cbar = plt.colorbar()
    cbar.ax.set_ylabel('Total Mass', fontsize='large')
    plt.savefig('/home/connor.mcisaac/public_html/imbh/mass_shifts_e{0}.png'.format(i + 1), bbox='tight')
    plt.close()

