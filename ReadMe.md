# Wavenet implementatio using Tensorflow tf.layers.conv1d
The goal of the repository is to provide an implementation of the WaveNet with Tensorflow middle level api(tf.layers.conv1d).

Based on https://github.com/ibab/tensorflow-wavenet



## Highlights

- tf.lyaers.conv1d
- fast generation algorithm(https://github.com/tomlepaine/fast-wavenet)
- We improved Fast wavenet implementation to filter_width >= 1 and batch_size >= 1  by using Queues.


```
FILE_PATTERN = r'NB([0-9])([0-9]+).([0-9]+)\.wav' 
```
in [audio_reader.py](https://github.com/hccho2/wavenet-tf.layers.conv1d/blob/master/wavenet/audio_reader.py).
must be modified according your audio file names.
