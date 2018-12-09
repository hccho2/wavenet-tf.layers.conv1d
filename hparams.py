# -*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np

hparams = tf.contrib.training.HParams(
    name = "WaveNet",
    batch_size = 1,
    store_metadata = False,
    histograms = False,
    learning_rate = 1e-3,              # Learning rate for training
    num_steps = 200000,                # Number of training steps
    
    optimizer = 'adam',
    momentum = 0.9,                   # 'Specify the momentum to be used by sgd or rmsprop optimizer. Ignored by the adam optimizer.
    max_checkpoints = 3,             # 'Maximum amount of checkpoints that will be kept alive. Default: '
    
    
    
    l2_regularization_strength = 0,  # Coefficient in the L2 regularization.
    sample_size = 100000,              # Concatenate and cut audio samples to this many samples
    silence_threshold = 0,             # Volume threshold below which to trim the start and the end from the training set samples.



    filter_width = 2,
    gc_channels = 32,                  # global_condition_vector의 차원. 이것 지정함으로써, global conditioning을 모델에 반영하라는 의미가 된다.
    sample_rate = 16000,
    dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512],
    residual_channels = 32,
    dilation_channels = 32,
    quantization_channels = 256,
    out_channels = 30,
    skip_channels = 512,
    use_biases = True,
    scalar_input = True,
    initial_filter_width = 32








)


def hparams_debug_string():
    values = hparams.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)
