#  coding: utf-8
"""Training script for the WaveNet network on the VCTK corpus.

This script trains a network with the WaveNet using data from the VCTK corpus,
which can be freely downloaded at the following site (~10 GB):
http://homepages.inf.ed.ac.uk/jyamagis/page3/page58/page58.html
"""

from __future__ import print_function

import argparse
from datetime import datetime
import json
import os
import sys
import time
from glob import glob
import tensorflow as tf
from tensorflow.python.client import timeline

from wavenet import WaveNetModel, AudioReader, optimizer_factory
tf.logging.set_verbosity(tf.logging.ERROR)
BATCH_SIZE = 1

DATA_DIRECTORY =  'D:\\hccho\\multi-speaker-tacotron-tensorflow-master\\datasets\son\\audio'   #   './VCTK-Corpus'

LOGDIR_ROOT = './logdir'


#LOGDIR = None
LOGDIR = './/logdir//train//2018-11-25T14-10-48'   # son
#LOGDIR = './/logdir//train//2018-11-08T21-09-51'   # test

GC_CHANNELS = 32 # gc_channels = embedding vector dim

CHECKPOINT_EVERY = 2000   # checkpoint 저장 주기
NUM_STEPS = 200000  # 최대 step
LEARNING_RATE = 1e-3
WAVENET_PARAMS = './wavenet_params.json'
STARTED_DATESTRING = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
SAMPLE_SIZE = 100000
L2_REGULARIZATION_STRENGTH = 0
SILENCE_THRESHOLD = 0
EPSILON = 0.001
MOMENTUM = 0.9
MAX_TO_KEEP = 3 # 최대 5개의 checkpoint만 저장되도록 
METADATA = False


def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    parser = argparse.ArgumentParser(description='WaveNet example network')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE,
                        help='How many wav files to process at once. Default: ' + str(BATCH_SIZE) + '.')
    parser.add_argument('--data_dir', type=str, default=DATA_DIRECTORY, help='The directory containing the VCTK corpus.')
    parser.add_argument('--store_metadata', type=bool, default=METADATA,help='Whether to store advanced debugging information (execution time, memory consumption) for use with '
                        'TensorBoard. Default: ' + str(METADATA) + '.')
    parser.add_argument('--logdir', type=str, default=LOGDIR,
                        help='Directory in which to store the logging information for TensorBoard. '
                        'If the model already exists, it will restore the state and will continue training. '
                        'Cannot use with --logdir_root and --restore_from.')
    parser.add_argument('--logdir_root', type=str, default=None,
                        help='Root directory to place the logging output and generated model. These are stored '
                        'under the dated subdirectory of --logdir_root. Cannot use with --logdir.')
    parser.add_argument('--restore_from', type=str, default=None,
                        help='Directory in which to restore the model from. This creates the new model under the dated directory '
                        'in --logdir_root. Cannot use with --logdir.')
    parser.add_argument('--checkpoint_every', type=int, default=CHECKPOINT_EVERY,help='How many steps to save each checkpoint after. Default: ' + str(CHECKPOINT_EVERY) + '.')
    parser.add_argument('--num_steps', type=int, default=NUM_STEPS,help='Number of training steps. Default: ' + str(NUM_STEPS) + '.')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE,help='Learning rate for training. Default: ' + str(LEARNING_RATE) + '.')
    parser.add_argument('--wavenet_params', type=str, default=WAVENET_PARAMS,help='JSON file with the network parameters. Default: ' + WAVENET_PARAMS + '.')
    parser.add_argument('--sample_size', type=int, default=SAMPLE_SIZE, help='Concatenate and cut audio samples to this many samples. Default: ' + str(SAMPLE_SIZE) + '.')
    parser.add_argument('--l2_regularization_strength', type=float,default=L2_REGULARIZATION_STRENGTH,
                        help='Coefficient in the L2 regularization. Default: False')
    parser.add_argument('--silence_threshold', type=float, default=SILENCE_THRESHOLD,
                        help='Volume threshold below which to trim the start and the end from the training set samples. Default: ' + str(SILENCE_THRESHOLD) + '.')
    parser.add_argument('--optimizer', type=str, default='adam', choices=optimizer_factory.keys(), help='Select the optimizer specified by this option. Default: adam.')
    parser.add_argument('--momentum', type=float,default=MOMENTUM, help='Specify the momentum to be used by sgd or rmsprop optimizer. Ignored by the '
                        'adam optimizer. Default: ' + str(MOMENTUM) + '.')
    parser.add_argument('--histograms', type=_str_to_bool, default=False,help='Whether to store histogram summaries. Default: False')
    
    # global_condition_vector의 차원. 이것 지정함으로써, global conditioning을 모델에 반영하라는 의미가 된다.
    parser.add_argument('--gc_channels', type=int, default=GC_CHANNELS, help='Number of global condition channels. Default: None. Expecting: Int')
    parser.add_argument('--max_checkpoints', type=int, default=MAX_TO_KEEP, help='Maximum amount of checkpoints that will be kept alive. Default: '   + str(MAX_TO_KEEP) + '.')
    return parser.parse_args()


def save(saver, sess, logdir, step):
    model_name = 'model.ckpt'
    checkpoint_path = os.path.join(logdir, model_name)
    print('Storing checkpoint to {} ...'.format(logdir), end="")
    sys.stdout.flush()

    if not os.path.exists(logdir):
        os.makedirs(logdir)

    saver.save(sess, checkpoint_path, global_step=step)
    print(' Done.')


def load(saver, sess, logdir):
    print("Trying to restore saved checkpoints from {} ...".format(logdir),end="")

    ckpt = tf.train.get_checkpoint_state(logdir)
    #ckpt = get_most_recent_checkpoint(logdir)
    if ckpt:
        print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
        global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1])
        print("  Global step was: {}".format(global_step))
        print("  Restoring...", end="")
        saver.restore(sess, ckpt.model_checkpoint_path)
        print(" Done.")
        return global_step
    else:
        print(" No checkpoint found.")
        return None


def get_default_logdir(logdir_root):
    logdir = os.path.join(logdir_root, 'train', STARTED_DATESTRING)
    if not os.path.exists(logdir):
        os.makedirs(logdir)    
    return logdir


def validate_directories(args):
    """Validate and arrange directory related arguments."""

    # Validation
    if args.logdir and args.logdir_root:
        raise ValueError("--logdir and --logdir_root cannot be "
                         "specified at the same time.")

    if args.logdir and args.restore_from:
        raise ValueError(
            "--logdir and --restore_from cannot be specified at the same "
            "time. This is to keep your previous model from unexpected "
            "overwrites.\n"
            "Use --logdir_root to specify the root of the directory which "
            "will be automatically created with current date and time, or use "
            "only --logdir to just continue the training from the last "
            "checkpoint.")

    # Arrangement
    logdir_root = args.logdir_root
    if logdir_root is None:
        logdir_root = LOGDIR_ROOT

    logdir = args.logdir
    if logdir is None:
        logdir = get_default_logdir(logdir_root)
        print('Using default logdir: {}'.format(logdir))

    restore_from = args.restore_from
    if restore_from is None:
        # args.logdir and args.restore_from are exclusive,
        # so it is guaranteed the logdir here is newly created.
        restore_from = logdir

    return {
        'logdir': logdir,
        'logdir_root': args.logdir_root,
        'restore_from': restore_from
    }

def get_most_recent_checkpoint(checkpoint_dir):
    checkpoint_paths = [path for path in glob("{}/*.ckpt-*.data-*".format(checkpoint_dir))]
    idxes = [int(os.path.basename(path).split('-')[1].split('.')[0]) for path in checkpoint_paths]

    max_idx = max(idxes)
    lastest_checkpoint = os.path.join(checkpoint_dir, "model.ckpt-{}".format(max_idx))

    #latest_checkpoint=checkpoint_paths[0]
    print(" [*] Found lastest checkpoint: {}".format(lastest_checkpoint))
    return lastest_checkpoint
def main():
    args = get_arguments()

    try:
        directories = validate_directories(args)
    except ValueError as e:
        print("Some arguments are wrong:")
        print(str(e))
        return

    logdir = directories['logdir']
    restore_from = directories['restore_from']

    # Even if we restored the model, we will treat it as new training
    # if the trained model is written into an arbitrary location.
    is_overwritten_training = logdir != restore_from

    with open(args.wavenet_params, 'r') as f:
        wavenet_params = json.load(f)

    # Create coordinator.
    coord = tf.train.Coordinator()

    # Load raw waveform from VCTK corpus.
    with tf.name_scope('create_inputs'):
        # Allow silence trimming to be skipped by specifying a threshold near
        # zero.
        silence_threshold = args.silence_threshold if args.silence_threshold > EPSILON else None
        gc_enabled = args.gc_channels is not None
        
        # AudioReader에서 wav 파일을 잘라 input값을 만든다. receptive_field길이만큼을 앞부분에 pad하거나 앞조각에서 가져온다. (receptive_field+ sample_size)크기로 자른다.
        reader = AudioReader(args.data_dir,coord,sample_rate=wavenet_params['sample_rate'],gc_enabled=gc_enabled,
                                receptive_field=WaveNetModel.calculate_receptive_field(wavenet_params["filter_width"], wavenet_params["dilations"],wavenet_params["scalar_input"], wavenet_params["initial_filter_width"]),
                                sample_size=args.sample_size,silence_threshold=silence_threshold)
        audio_batch = reader.dequeue(args.batch_size)  # (batch_size, ?, 1)
        if gc_enabled:
            gc_id_batch = reader.dequeue_gc(args.batch_size) # [1,2,2,1,...] <--- batch size 길이
        else:
            gc_id_batch = None

    # Create network.
    net = WaveNetModel(
        batch_size=args.batch_size,
        dilations=wavenet_params['dilations'],
        filter_width=wavenet_params['filter_width'],
        residual_channels=wavenet_params['residual_channels'],
        dilation_channels=wavenet_params['dilation_channels'],
        quantization_channels=wavenet_params['quantization_channels'],
        skip_channels=wavenet_params['skip_channels'],
        use_biases=wavenet_params['use_biases'],  #  True
        scalar_input=wavenet_params['scalar_input'],
        initial_filter_width=wavenet_params['initial_filter_width'],
        histograms=args.histograms,
        global_condition_channels=args.gc_channels,
        global_condition_cardinality=reader.gc_category_cardinality,
        train_mode=True)

    if args.l2_regularization_strength == 0:
        args.l2_regularization_strength = None
        
       
    loss = net.loss(input_batch=audio_batch, global_condition_batch=gc_id_batch, l2_regularization_strength=args.l2_regularization_strength)
     
    optimizer = optimizer_factory[args.optimizer](learning_rate=args.learning_rate,momentum=args.momentum)
    
    trainable = tf.trainable_variables()
    
    optim = optimizer.minimize(loss, var_list=trainable)

    run_metadata = tf.RunMetadata()

    # Set up session
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))  # log_device_placement=False --> cpu/gpu 자동 배치.
    init = tf.global_variables_initializer()
    sess.run(init)

    # Saver for storing checkpoints of the model.
    saver = tf.train.Saver(var_list=tf.global_variables(), max_to_keep=args.max_checkpoints)  # 최대 checkpoint 저장 갯수 지정

    try:
        saved_global_step = load(saver, sess, restore_from)  # checkpoint load
        if is_overwritten_training or saved_global_step is None:
            # The first training step will be saved_global_step + 1,
            # therefore we put -1 here for new or overwritten trainings.
            saved_global_step = -1

    except:
        print("Something went wrong while restoring checkpoint. We will terminate training to avoid accidentally overwriting the previous model.")
        raise

    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    reader.start_threads(sess)

    step = None
    last_saved_step = saved_global_step
    try:
        for step in range(saved_global_step + 1, args.num_steps+1):
            start_time = time.time()
            if args.store_metadata and step % 50 == 0:
                # Slow run that stores extra information for debugging.
                print('Storing metadata')
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                loss_value, _ = sess.run([loss, optim],options=run_options,run_metadata=run_metadata)

                tl = timeline.Timeline(run_metadata.step_stats)
                timeline_path = os.path.join(logdir, 'timeline.trace')
                with open(timeline_path, 'w') as f:
                    f.write(tl.generate_chrome_trace_format(show_memory=True))
            else:
                loss_value, _ = sess.run([loss, optim])

            duration = time.time() - start_time
            print('step {:d} - loss = {:.3f}, ({:.3f} sec/step)'.format(step, loss_value, duration))

            if step % args.checkpoint_every == 0:
                save(saver, sess, logdir, step)
                last_saved_step = step

    except KeyboardInterrupt:
        # Introduce a line break after ^C is displayed so save message
        # is on its own line.
        print()
    finally:
        if step > last_saved_step:
            save(saver, sess, logdir, step)
        coord.request_stop()
        coord.join(threads)


if __name__ == '__main__':
    main()
    print('Done')
