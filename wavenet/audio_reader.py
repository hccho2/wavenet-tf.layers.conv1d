#  coding: utf-8
import fnmatch
import os
import random
import re
import threading

import librosa
import numpy as np
import tensorflow as tf

FILE_PATTERN = r'NB([0-9])([0-9]+).([0-9]+)\.wav'  #r'LJ([0-9]+)-([0-9]+)\.wav'      r'p([0-9]+)_([0-9]+)\.wav'     # p227_363.wav


def get_category_cardinality(files):
    id_reg_expression = re.compile(FILE_PATTERN)
    min_id = None
    max_id = None
    for filename in files:
        matches = id_reg_expression.findall(filename)[0]  # [('227', '363')]  --> ('227', '363')
        id, *_ = [int(id_) for id_ in matches]
        if min_id is None or id < min_id:
            min_id = id
        if max_id is None or id > max_id:
            max_id = id

    return min_id, max_id


def randomize_files(files):
    for file in files:
        file_index = random.randint(0, (len(files) - 1))   # ranodm 정수 1개 return
        yield files[file_index]


def find_files(directory, pattern='*.wav'):
    '''Recursively finds all files matching the pattern.'''
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files


def load_generic_audio(directory, sample_rate):
    '''Generator that yields audio waveforms from the directory.'''
    files = find_files(directory)
    id_reg_exp = re.compile(FILE_PATTERN)
    print("files length: {}".format(len(files)))
    randomized_files = randomize_files(files)
    for filename in randomized_files:
        ids = id_reg_exp.findall(filename)
        if not ids:
            # The file name does not match the pattern containing ids, so
            # there is no id.
            category_id = None
        else:
            # The file name matches the pattern for containing ids.
            category_id = int(ids[0][0])
        audio, _ = librosa.load(filename, sr=sample_rate, mono=True)
        audio = audio.reshape(-1, 1)
        yield audio, filename, category_id


def trim_silence(audio, threshold, frame_length=2048):
    '''Removes silence at the beginning and end of a sample.'''
    if audio.size < frame_length:
        frame_length = audio.size
    energy = librosa.feature.rmse(audio, frame_length=frame_length)
    frames = np.nonzero(energy > threshold)
    indices = librosa.core.frames_to_samples(frames)[1]

    # Note: indices can be an empty array, if the whole audio was silence.
    return audio[indices[0]:indices[-1]] if indices.size else audio[0:0]


def not_all_have_id(files):
    ''' Return true iff any of the filenames does not conform to the pattern
        we require for determining the category id.'''
    id_reg_exp = re.compile(FILE_PATTERN)
    for file in files:
        ids = id_reg_exp.findall(file)   # 'p227_363.wav' --> [('227', '363')]
        if not ids:
            return True
    return False


class AudioReader(object):
    '''Generic background audio reader that preprocesses audio files
    and enqueues them into a TensorFlow queue.'''

    def __init__(self,
                 audio_dir,
                 coord,
                 sample_rate,
                 gc_enabled,
                 receptive_field,
                 sample_size=None,
                 silence_threshold=None,
                 queue_size=32):
        self.audio_dir = audio_dir
        self.sample_rate = sample_rate
        self.coord = coord
        self.sample_size = sample_size
        self.receptive_field = receptive_field
        self.silence_threshold = silence_threshold
        self.gc_enabled = gc_enabled
        self.threads = []
        
        
        
        """
        ###
        self.sample_placeholder = tf.placeholder(dtype=tf.float32, shape=None)  # waveform
        self.queue = tf.PaddingFIFOQueue(queue_size, ['float32'], shapes=[(None, 1)]) # A FIFOQueue that supports batching variable-sized tensors by padding
        self.enqueue = self.queue.enqueue([self.sample_placeholder])

        if self.gc_enabled:
            self.id_placeholder = tf.placeholder(dtype=tf.int32, shape=())
            self.gc_queue = tf.PaddingFIFOQueue(queue_size, ['int32'], shapes=[()])
            self.gc_enqueue = self.gc_queue.enqueue([self.id_placeholder])
        ###    
        """    
            
        self.sample_placeholder = [tf.placeholder(tf.float32, shape=None)]          
            
        if self.gc_enabled:
            self.sample_placeholder.append(tf.placeholder(tf.int32, shape=None))            
            self.queue = tf.PaddingFIFOQueue(queue_size, [tf.float32, tf.int32], shapes=[(None, 1), ()], name='input_queue')
        else:
            self.queue = tf.PaddingFIFOQueue(queue_size, [tf.float32], shapes=[(None, 1)],name='input_queue')
        self.enqueue = self.queue.enqueue(self.sample_placeholder)    
            
            
            
            

        # TODO Find a better way to check this.
        # Checking inside the AudioReader's thread makes it hard to terminate
        # the execution of the script, so we do it in the constructor for now.
        files = find_files(audio_dir)  # ['.//TEST-Corpus\\LJ001-0001.wav', './/TEST-Corpus\\LJ001-0002.wav', ...]
        if not files:
            raise ValueError("No audio files found in '{}'.".format(audio_dir))
        
        
        # filename에 global condition에 대한 정보가 있는지 check
        if self.gc_enabled and not_all_have_id(files):
            raise ValueError("Global conditioning is enabled, but file names do not conform to pattern having id.")
        
        
        # Determine the number of mutually-exclusive categories we will accomodate in our embedding table.
        if self.gc_enabled:
            _, self.gc_category_cardinality = get_category_cardinality(files)
            # Add one to the largest index to get the number of categories,
            # since tf.nn.embedding_lookup expects zero-indexing. This
            # means one or more at the bottom correspond to unused entries
            # in the embedding lookup table. But that's a small waste of memory
            # to keep the code simpler, and preserves correspondance between
            # the id one specifies when generating, and the ids in the
            # file names.
            self.gc_category_cardinality += 1   # index가 0부터 시작할 수도 있으나, 1개 더해준다. 낭비일 수도 있다.  --> gc_category_cardinality는 gc의 갯수는 아니다. 그냥 파일명에서 가장 높은 수 + 1
            print("Detected --gc_cardinality={}".format(self.gc_category_cardinality))
        else:
            self.gc_category_cardinality = None

    def dequeue(self, num_elements):  # num_elements <--- batch_size를 의미함.
        output = self.queue.dequeue_many(num_elements)
        return output


    def thread_main(self, sess):
        # 여기서 self.enqueue와 self.gc_enqueue를 같이 처리하기 때문에 pair가 맞아진다.
        stop = False
        # Go through the dataset multiple times
        while not stop:
            iterator = load_generic_audio(self.audio_dir, self.sample_rate)  # audio files들에 대한 정보와 librosa.load를 실행한다.
            for audio, filename, category_id in iterator:   # audio: (117329, 1), filename = './/TEST-Corpus\\LJ001-0036.wav', category_id = None
                if self.coord.should_stop():
                    stop = True
                    break
                if self.silence_threshold is not None:
                    # Remove silence
                    audio = trim_silence(audio[:, 0], self.silence_threshold)
                    audio = audio.reshape(-1, 1)
                    if audio.size == 0:
                        print("Warning: {} was ignored as it contains only "
                              "silence. Consider decreasing trim_silence "
                              "threshold, or adjust volume of the audio."
                              .format(filename))

                # receptive_field 크기의 padding을 앞에 붙힌다.
                audio = np.pad(audio, [[self.receptive_field, 0], [0, 0]],'constant')  # (117329, 1) ==> (5117+117329, 1)

                if self.sample_size:
                    # Cut samples into pieces of size receptive_field + sample_size with receptive_field overlap
                    while len(audio) > self.receptive_field:
                        piece = audio[:(self.receptive_field + self.sample_size), :]
                        
                        audio = audio[self.sample_size:, :]
                        if self.gc_enabled:
                            sess.run(self.enqueue, feed_dict= dict(zip(self.sample_placeholder,(piece,category_id))))   #category_id는 파일에서 읽어들인 speaker번호
                        else:
                            sess.run(self.enqueue, feed_dict= dict(zip(self.sample_placeholder,(piece))))
                else:
                    # sample_size가 지정되지 않으면, receptive_field크기의 padding만 붙혀 통으로 넘긴다.
                    if self.gc_enabled:
                        sess.run(self.enqueue, feed_dict= dict(zip(self.sample_placeholder,(audio,category_id))))
                    else:
                        sess.run(self.enqueue, feed_dict= dict(zip(self.sample_placeholder,(audio))))

    def start_threads(self, sess, n_threads=1):
        for _ in range(n_threads):
            thread = threading.Thread(target=self.thread_main, args=(sess,))
            thread.daemon = True  # Thread will close when parent quits.
            thread.start()
            self.threads.append(thread)
        return self.threads
