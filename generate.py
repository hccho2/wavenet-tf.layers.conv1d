#  coding: utf-8
"""
sample_rate = 16000이므로, samples 48000이면 3초 길이가 된다.

> python generate.py --samples 48000 --gc_cardinality 2 --gc_id 1 ./logdir/train/2018-12-21T10-22-59
> python generate.py --samples 16000 --wav_seed ./logdir/seed.wav --gc_cardinality 2 --gc_id 1 ./logdir/train/2018-12-21T10-22-59

wav_seed를 줄때는 samples 갯수보다 길어야 한다.
> python generate.py --samples 48000 --wav_seed ./logdir/seed.wav ./logdir/train/2018-11-01T22-46-56/model.ckpt-102000


"""
import argparse
from datetime import datetime
import json
import os,time

import librosa
import numpy as np
import tensorflow as tf

from wavenet import WaveNetModel, mu_law_decode, mu_law_encode, audio_reader
from tensorboard.summary import audio
from hparams import hparams
from utils import load_hparams,load

SAMPLES = 16000
TEMPERATURE = 1.0
LOGDIR = './logdir'
SILENCE_THRESHOLD = 0  # 0.1
GC_CHANNELS = 32 # gc_channels = embedding vector dim
BATCH_SIZE = 2
def get_arguments():
    def _str_to_bool(s):
        """Convert string to bool (in argparse context)."""
        if s.lower() not in ['true', 'false']:
            raise ValueError('Argument needs to be a boolean, got {}'.format(s))
        return {'true': True, 'false': False}[s.lower()]

    def _ensure_positive_float(f):
        """Ensure argument is a positive float."""
        if float(f) < 0:
            raise argparse.ArgumentTypeError('Argument must be greater than zero')
        return float(f)

    parser = argparse.ArgumentParser(description='WaveNet generation script')
    parser.add_argument('checkpoint_dir', type=str, help='Which model checkpoint to generate from')
    parser.add_argument('--samples',type=int,default=SAMPLES, help='How many waveform samples to generate')
    parser.add_argument('--temperature', type=_ensure_positive_float, default=TEMPERATURE,help='Sampling temperature')
    parser.add_argument('--logdir',type=str,default=LOGDIR,help='Directory in which to store the logging information for TensorBoard.')
    parser.add_argument('--wav_out_path',type=str,default=None,help='Path to output wav file')

    
    
    parser.add_argument('--wav_seed',type=str,default=None,help='The wav file to start generation from')
    parser.add_argument('--gc_channels',type=int,default=GC_CHANNELS,help='Number of global condition embedding channels. Omit if no global conditioning.')
    parser.add_argument('--gc_cardinality',type=int,default=None,help='Number of categories upon which we globally condition.')
    parser.add_argument('--gc_id',type=int,default=None,help='ID of category to generate, if globally conditioned.')
    
    arguments = parser.parse_args()
    if arguments.gc_channels is not None:
        if arguments.gc_cardinality is None:
            raise ValueError("Globally conditioning but gc_cardinality not specified. Use --gc_cardinality=377 for full VCTK corpus.")

        if arguments.gc_id is None:
            raise ValueError("Globally conditioning, but global condition was not specified. Use --gc_id to specify global condition.")

    return arguments


def write_wav(waveform, sample_rate, filename):
    y = np.array(waveform)
    librosa.output.write_wav(filename, y, sample_rate)
    print('Updated wav file at {}'.format(filename))


def create_seed(filename,sample_rate,quantization_channels,window_size,scalar_input,silence_threshold=SILENCE_THRESHOLD):
    audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio_reader.trim_silence(audio, silence_threshold)
    if scalar_input:
        if len(audio) < window_size:
            return audio
        else: return audio[:window_size]
    else:
        quantized = mu_law_encode(audio, quantization_channels)
    
    
        # 짧으면 짧은 대로 return하는데, padding이라도 해야되지 않나???
        cut_index = tf.cond(tf.size(quantized) < tf.constant(window_size), lambda: tf.size(quantized), lambda: tf.constant(window_size))
    
        return quantized[:cut_index]


def main():
    config = get_arguments()
    started_datestring = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())
    logdir = os.path.join(config.logdir, 'generate', started_datestring)
    
    if not os.path.exists(logdir):
        os.makedirs(logdir)


    load_hparams(hparams, config.checkpoint_dir)






    with tf.device('/cpu:0'):

        sess = tf.Session()
        scalar_input = hparams.scalar_input
        net = WaveNetModel(
            batch_size=BATCH_SIZE,
            dilations=hparams.dilations,
            filter_width=hparams.filter_width,
            residual_channels=hparams.residual_channels,
            dilation_channels=hparams.dilation_channels,
            quantization_channels=hparams.quantization_channels,
            out_channels =hparams.out_channels,
            skip_channels=hparams.skip_channels,
            use_biases=hparams.use_biases,
            scalar_input=scalar_input,
            initial_filter_width=hparams.initial_filter_width,
            global_condition_channels=hparams.gc_channels,
            global_condition_cardinality=config.gc_cardinality,train_mode=False)   # train 단계에서는 global_condition_cardinality를 AudioReader에서 파악했지만, 여기서는 넣어주어야 함
    
        if scalar_input:
            samples = tf.placeholder(tf.float32,shape=[net.batch_size,None])
        else:
            samples = tf.placeholder(tf.int32,shape=[net.batch_size,None])  # samples: mu_law_encode로 변환된 것. one-hot으로 변환되기 전. (batch_size, 길이)
    

        next_sample = net.predict_proba_incremental(samples, [config.gc_id]*net.batch_size)  # Fast Wavenet Generation Algorithm-1611.09482 algorithm 적용
        
            
        var_list = [var for var in tf.global_variables() if 'queue' not in var.name ]
        saver = tf.train.Saver(var_list)
        print('Restoring model from {}'.format(config.checkpoint_dir))
        
        load(saver, sess, config.checkpoint_dir)
        
        sess.run(net.queue_initializer) # 이 부분이 없으면, checkpoint에서 복원된 값들이 들어 있다.
    
        
    
        quantization_channels = hparams.quantization_channels
        if config.wav_seed:
            # wav_seed의 길이가 receptive_field보다 작으면, padding이라도 해야 되는 거 아닌가? 그냥 짧으면 짧은 대로 return함  --> 그래서 너무 짧으면 error
            seed = create_seed(config.wav_seed,hparams.sample_rate,quantization_channels,net.receptive_field,scalar_input)  # --> mu_law encode 된 것.
            if scalar_input:
                waveform = seed.tolist()
            else:
                waveform = sess.run(seed).tolist()  # [116, 114, 120, 121, 127, ...]

            print('Priming generation...')
            for i, x in enumerate(waveform[-net.receptive_field: -1]):  # 제일 마지막 1개는 아래의 for loop의 첫 loop에서 넣어준다.
                if i % 100 == 0:
                    print('Priming sample {}'.format(i))
                sess.run(next_sample, feed_dict={samples: np.array([x]*net.batch_size).reshape(net.batch_size,1)})
            print('Done.')
            waveform = np.array([waveform[-net.receptive_field:]]*net.batch_size)            
        else:
            # Silence with a single random sample at the end.
            if scalar_input:
                waveform = [0.0] * (net.receptive_field - 1)
                waveform = np.array(waveform*net.batch_size).reshape(net.batch_size,-1)
                waveform = np.concatenate([waveform,2*np.random.rand(net.batch_size).reshape(net.batch_size,-1)-1],axis=-1) # -1~1사이의 random number를 만들어 끝에 붙힌다.
            else:
                waveform = [quantization_channels / 2] * (net.receptive_field - 1)  # 필요한 receptive_field 크기보다 1개 작게 만든 후, 아래에서 random하게 1개를 덧붙힌다.
                waveform = np.array(waveform*net.batch_size).reshape(net.batch_size,-1)
                waveform = np.concatenate([waveform,np.random.randint(quantization_channels,size=net.batch_size).reshape(net.batch_size,-1)],axis=-1)  # one hot 변환 전. (batch_size, 5117)
            
    
        last_sample_timestamp = datetime.now()
        for step in range(config.samples):  # 원하는 길이를 구하기 위해 loop 

            window = waveform[:,-1:]  # 제일 끝에 있는 1개만 samples에 넣어 준다.

    
            # Run the WaveNet to predict the next sample.
            
            # fast가 아닌경우. window: [128.0, 128.0, ..., 128.0, 178, 185]
            # fast인 경우, window는 숫자 1개.
            prediction = sess.run(next_sample, feed_dict={samples: window})  # samples는 mu law encoding된 것. 계산 과정에서 one hot으로 변환된다.  --> (batch_size,256)
    
            if scalar_input:
                sample = prediction
            else:
                # Scale prediction distribution using temperature.
                # 다음 과정은 config.temperature==1이면 각 원소를 합으로 나누어주는 것에 불과. 이미 softmax를 적용한 겂이므로, 합이 1이된다. 그래서 값의 변화가 없다.
                # config.temperature가 1이 아니며, 각 원소의 log취한 값을 나눈 후, 합이 1이 되도록 rescaling하는 것이 된다.
                np.seterr(divide='ignore')
                scaled_prediction = np.log(prediction) / config.temperature   # config.temperature인 경우는 값의 변화가 없다.
                scaled_prediction = (scaled_prediction - np.logaddexp.reduce(scaled_prediction,axis=-1,keepdims=True))  # np.log(np.sum(np.exp(scaled_prediction)))
                scaled_prediction = np.exp(scaled_prediction)
                np.seterr(divide='warn')
        
                # Prediction distribution at temperature=1.0 should be unchanged after
                # scaling.
                if config.temperature == 1.0:
                    np.testing.assert_allclose( prediction, scaled_prediction, atol=1e-5, err_msg='Prediction scaling at temperature=1.0 is not working as intended.')
    
                sample = [[np.random.choice(np.arange(quantization_channels), p=p)] for p in scaled_prediction]  # choose one sample per batch
            
            waveform = np.concatenate([waveform,sample],axis=-1)
    
            # Show progress only once per second.
            current_sample_timestamp = datetime.now()
            time_since_print = current_sample_timestamp - last_sample_timestamp
            if time_since_print.total_seconds() > 1.:
                print('Sample {:3<d}/{:3<d}'.format(step + 1, config.samples), end='\r')
                last_sample_timestamp = current_sample_timestamp
    
    
        # Introduce a newline to clear the carriage return from the progress.
        print()
    
        
        # Save the result as a wav file.    
        if scalar_input:
            out = waveform
        else:
            decode = mu_law_decode(samples, quantization_channels)
            out = sess.run(decode, feed_dict={samples: waveform})
        for i in range(net.batch_size):
            config.wav_out_path= logdir + '/test-{}.wav'.format(i)
            write_wav(out[i], hparams.sample_rate, config.wav_out_path)
        
        print('Finished generating.')


if __name__ == '__main__':
    s = time.time()
    main()
    print(time.time()-s,'sec')