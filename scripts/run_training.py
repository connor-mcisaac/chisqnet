#!/home/connor.mcisaac/envs/tensor-imbh/bin/python3.6

import os
import numpy as np
import argparse
import logging
from preprocessing import TriggerList, InjectionTriggers, StrainGen, TemplateGen, BatchGen
from filtering import MatchedFilter, ShiftTransform
import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


parser = argparse.ArgumentParser()
parser.add_argument("--ifos", nargs='+', required=True)
parser.add_argument("--frames", nargs='+', required=True)
parser.add_argument("--channels", nargs='+', required=True)
parser.add_argument("--trigger-files", nargs='+', required=True)
parser.add_argument("--injection-dirs", nargs='+', required=True)
parser.add_argument("--injection-approximants", nargs='+', required=True)
parser.add_argument("--segment-files", nargs='+', required=True)
parser.add_argument("--foreground-vetos", nargs='+', required=True)
parser.add_argument("--bank", required=True)
parser.add_argument("--trigger-output-file", required=True)
parser.add_argument("--injection-output-file", required=True)
parser.add_argument("--output-file", required=True)
parser.add_argument("--sample-rate", type=int, default=2048)
parser.add_argument("--data-width", type=float, default=512.)
parser.add_argument("--cut-width", type=float, default=16.)
parser.add_argument("--snr-width", type=float, default=0.2)
parser.add_argument("--verbose", action='store_true')
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--batch-num", type=int, default=1)
parser.add_argument("--shift-num", type=int, default=8)
args = parser.parse_args()

if args.verbose:
    log_level = logging.DEBUG
else:
    log_level = logging.WARNING
logging.basicConfig(format='%(asctime)s : %(message)s', level=log_level)

if not os.path.isfile(args.trigger_output_file):
    logging.info("Reading triggers from pycbc workflow")
    triggers = TriggerList(args.trigger_files)
    triggers.threshold_cut(8, 'snr')
    triggers.get_newsnr()
    triggers.threshold_cut(8, 'newsnr_sg')
    triggers.get_bank_params(args.bank)
    triggers.cluster_over_time(1.0, 'newsnr_sg')
    triggers.apply_segments(args.segment_files, "TRIGGERS_GENERATED")
    triggers.apply_segments(args.foreground_vetos, "closed_box", within=False)
    logging.info("Saving triggers to {0}".format(args.trigger_output_file))
    triggers.write_to_hdf(args.trigger_output_file)
    logging.info("Triggers saved")
else:
    logging.info("Loading triggers from {0}".format(args.trigger_output_file))
    triggers = TriggerList.read_from_hdf(args.trigger_output_file)

if not os.path.isfile(args.injection_output_file):
    logging.info("Reading injections from pycbc workflow")
    injections = InjectionTriggers(args.injection_dirs, args.injection_approximants)
    injections.threshold_cut(6, 'snr')
    injections.get_bank_params(args.bank)
    #injections.get_newsnr()
    #injections.threshold_cut(6, 'newsnr_sg')
    #injections.cluster_over_time(1.0, 'newsnr_sg')
    injections.apply_segments(args.segment_files, "TRIGGERS_GENERATED")
    injections.apply_segments(args.foreground_vetos, "closed_box", within=False)
    logging.info("Saving injections to {0}".format(args.injection_output_file))
    injections.write_to_hdf(args.injection_output_file)
    logging.info("Injections saved")
else:
    logging.info("Loading injections from {0}".format(args.injection_output_file))
    injections = InjectionTriggers.read_from_hdf(args.injection_output_file)

logging.info("{0} noise triggers available".format(len(triggers.flatten())))
logging.info("{0} injections available".format(len(injections.flatten())))

gens = {}
for ifo, frame, channel in zip(args.ifos, args.frames, args.channels):
   gens[ifo] = StrainGen(ifo, args.data_width, frame, channel, args.segment_files,
                         sample_rate=args.sample_rate, start_pad=32., end_pad=16.)

test = triggers.draw_triggers(ifo=args.ifos[0])
strain = gens[args.ifos[0]].get_strain(test['end_time'][0])
cut_strain, cut_stilde = gens[args.ifos[0]].get_cut(test['end_time'][0], args.cut_width)

freqs = cut_stilde.sample_frequencies
f_high = freqs[-1]
delta_f = freqs[1] - freqs[0]
length = len(freqs)

templates = TemplateGen(10., delta_f, length)

batch = BatchGen(gens, triggers, injections, templates, args.batch_size,
                 cut_width=args.cut_width, snr_width=args.snr_width)

match = MatchedFilter(freqs, 15., f_high)

transform = ShiftTransform(args.shift_num, 2, 1e-2, 1e1, 500,
                           freqs, 15., f_high)

trainables = [transform.dts, transform.dfs]

chi2 = tfp.distributions.Chi2(tf.constant(1., dtype=tf.float64))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-8)

for i in range(args.batch_num):
    logging.info("Starting batch {0}".format(i))
    start = time.time()

    data, psd, temp, params, label, cut_idxs = batch.get_batch()

    data = tf.convert_to_tensor(data, dtype=tf.complex128)
    psd = tf.convert_to_tensor(psd, dtype=tf.complex128)
    temp = tf.convert_to_tensor(temp, dtype=tf.complex128)
    label = tf.convert_to_tensor(label, dtype=tf.bool)
    
    mass = tf.convert_to_tensor(params["mass1"] + params["mass2"], dtype=tf.float64)

    logging.info("Batch inputs read")

    with tf.GradientTape() as tape:

        dts = transform.get_dt(mass)
        dfs = transform.get_df(mass)

        chi_temps = transform.shift_dt_df(temp, mass)
        logging.info("Templates shifted")

        chi_orthos =  transform.get_ortho(chi_temps, temp, psd)
        logging.info("Orthogonal templates created")

        temp = tf.expand_dims(temp, axis=1)
        temps = tf.concat([temp, chi_orthos], axis=1)
        
        snrs = match(temps, data, psd)
        snrs = tf.math.abs(snrs)
        
        snr = snrs[:, 0, :]
        chis = snrs[:, 1:, :]

        logging.info("SNRs calculated")

        chisq = tf.math.reduce_sum(chis ** 2., axis=1) / 2. / args.shift_num

        chisq_thresh = tf.math.maximum(chisq, tf.ones_like(chisq, dtype=tf.float64))

        snr_prime = snr / chisq_thresh ** 0.5

        chisq_cut = [chisq[j, cut_idxs[j, 0]:cut_idxs[j, 1]] for j in range(args.batch_size)]
        snr_cut = [snr[j, cut_idxs[j, 0]:cut_idxs[j, 1]] for j in range(args.batch_size)]
        snr_prime_cut = [snr_prime[j, cut_idxs[j, 0]:cut_idxs[j, 1]] for j in range(args.batch_size)]

        chisq_cut = tf.stack(chisq_cut)
        snr_cut = tf.stack(snr_cut)
        snr_prime_cut = tf.stack(snr_prime_cut)

        max_snr = tf.math.argmax(snr_cut, axis=-1)
        idx = tf.stack([tf.range(args.batch_size, dtype=tf.int64), max_snr], axis=-1)

        chi_max = tf.gather_nd(chisq_cut, idx)
        snr_max = tf.gather_nd(snr_cut, idx)
        snr_prime_max = tf.gather_nd(snr_prime_cut, idx)

        logging.info("SNR' calculated")

        snr_logp = - chi2.log_prob(snr_prime_max)
        chi_logp = - chi2.log_prob(chi_max)

        num_inj = tf.math.reduce_sum(tf.cast(label, tf.float64)).numpy()

        trig_weight = tf.ones_like(label, dtype=tf.float64)
        trig_weight /= (args.batch_size - num_inj)

        inj_weight = tf.ones_like(label, dtype=tf.float64)
        inj_num = tf.ones_like(label, dtype=tf.float64) * num_inj
        inj_weight /= tf.maximum(inj_weight, inj_num)

        batch_loss = tf.where(label,
                              chi_logp * inj_weight,
                              snr_logp * trig_weight)
        loss = tf.reduce_mean(batch_loss)

        logging.info("Loss calculated")

    gradients = tape.gradient(loss, trainables)
    gradients = [tf.where(tf.math.is_nan(grad), tf.zeros_like(grad), grad)
                 for grad in gradients]
    optimizer.apply_gradients(zip(gradients, trainables))

    logging.info("Gradients applied")
    logging.info("Completed batch {0}".format(i))

    duration = time.time() - start
    print("Batch: {0:5d}, Loss: {1:12.6f}, ".format(i, loss)
          + "Time: {0:8.2f}, Num injections: {1:3d}".format(duration, int(num_inj)))

snr = snr_cut.numpy()
chisq = chisq_cut.numpy()
snr_prime = snr_prime_cut.numpy()
times = 1. * np.arange(snr.shape[-1]) / args.sample_rate

snr_max = snr_max.numpy()
snr_prime_max = snr_prime_max.numpy()

label = label.numpy()

for i in range(args.batch_size):

    plt.figure(figsize=(16, 6))
    plt.plot(times, snr[i, :], alpha=0.75, label="snr")
    plt.plot(times, chisq[i, :], alpha=0.75, label="chisq")
    plt.plot(times, snr_prime[i, :], alpha=0.75, label="snr'")
    plt.grid()
    plt.legend()
    plt.title("injection = " + str(label[i]) + " SNR = " + str(snr_max[i])
              + " SNR' = " + str(snr_prime_max[i]))
    plt.savefig('/home/connor.mcisaac/public_html/imbh/' + str(i) + '_snr.png')

masses = tf.Variable(np.linspace(100, 500, num=17, endpoint=True))
dts = transform.get_dt(masses).numpy()
dfs = transform.get_df(masses).numpy()
masses = masses.numpy()

plt.figure(figsize=(8, 6))
for i in range(args.shift_num):
    plt.scatter(dts[:, i], dfs[:, i], c=masses, cmap="cool")
plt.grid()
plt.colorbar()
plt.savefig('/home/connor.mcisaac/public_html/imbh/mass_shifts.png')
