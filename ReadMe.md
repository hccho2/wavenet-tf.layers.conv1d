# Wavenet implementation using Tensorflow tf.layers.conv1d
The goal of the repository is to provide an implementation of the WaveNet with Tensorflow middle level api(tf.layers.conv1d).

Based on https://github.com/ibab/tensorflow-wavenet



## Highlights

- Dilation convolutions are implented by tf.layers.conv1d
- fast generation algorithm(https://github.com/tomlepaine/fast-wavenet)
- We improved Fast wavenet implementation to filter_width >= 1 and batch_size >= 1  by using Queues.
- Mixture of logistic distributions loss / sampling

```
FILE_PATTERN = r'NB([0-9])([0-9]+).([0-9]+)\.wav' 
```
FILE_PATTERN in [audio_reader.py](https://github.com/hccho2/wavenet-tf.layers.conv1d/blob/master/wavenet/audio_reader.py)
shoud be modified depending on your audio file names.
