#  coding: utf-8
import numpy as np
import tensorflow as tf

from .ops import causal_conv, mu_law_encode

class WaveNetModel(object):
    def __init__(self,batch_size,dilations,filter_width,residual_channels,dilation_channels,skip_channels,quantization_channels=2**8,
                 use_biases=False,scalar_input=False,initial_filter_width=32,histograms=False,global_condition_channels=None,
                 global_condition_cardinality=None,train_mode=True):

        self.batch_size = batch_size
        self.dilations = dilations
        self.filter_width = filter_width
        self.residual_channels = residual_channels
        self.dilation_channels = dilation_channels
        self.quantization_channels = quantization_channels
        self.use_biases = use_biases
        self.skip_channels = skip_channels
        self.scalar_input = scalar_input
        self.initial_filter_width = initial_filter_width
        self.histograms = histograms
        self.global_condition_channels = global_condition_channels
        self.global_condition_cardinality = global_condition_cardinality
        self.train_mode = train_mode
        self.receptive_field = WaveNetModel.calculate_receptive_field(self.filter_width, self.dilations, self.scalar_input,self.initial_filter_width)


    @staticmethod
    def calculate_receptive_field(filter_width, dilations, scalar_input, initial_filter_width):
        # causal 때문에 length (T-1) + (여기서 계산되는 receptive_field만큼의  padding)  --> 최종 output의 길이가 T가 된다.
        receptive_field = (filter_width - 1) * sum(dilations) + 1  # 마지막 +1은 causal condition 때문에 1개 자른 것의 때문에 길이가 T-1인 되기 때문에 +1을 통해서 입력과 같은 길이 T가 된다.
        if scalar_input:
            receptive_field += initial_filter_width - 1
        else:
            receptive_field += filter_width - 1  # dilation layer에 들어가지 전에 Causal Convolution을 한번 하기 때문에, dilation size = 1짜리가 하나 더 있는 것과 같은 효과
        return receptive_field

    def _create_causal_layer(self, input_batch):
        with tf.name_scope('causal_layer'):
            return tf.layers.conv1d(input_batch,filters=self.residual_channels,kernel_size=self.filter_width,padding='valid',dilation_rate=1,use_bias=False)


    def _create_queue(self):
        with tf.variable_scope('queue'):
            self.causal_queue = tf.Variable(initial_value=tf.zeros(shape=[self.batch_size,(self.filter_width),self.quantization_channels], dtype=tf.float32), name='causal_queue', trainable=False)
            
            self.dilation_queue=[]
            for i,d in enumerate(self.dilations):
                q = tf.Variable(initial_value=tf.zeros(shape=[self.batch_size,d*(self.filter_width-1)+1,self.dilation_channels], dtype=tf.float32), name='dilation_queue'.format(i), trainable=False)
                self.dilation_queue.append(q)
        
        # restore했을 때, Dilation_Queue,Causal_Queue는 0으로 initialization해야 한다.
        self.queue_initializer= tf.variables_initializer(self.dilation_queue + [self.causal_queue])

    def _create_dilation_layer(self, input_batch, layer_index, dilation,global_condition_batch, output_width):
        with tf.variable_scope('dilation_layer'):
            conv_filter = tf.layers.conv1d(input_batch,filters=self.dilation_channels,kernel_size=self.filter_width,dilation_rate=dilation,padding='valid',use_bias=False,name='conv_filter')
            conv_gate = tf.layers.conv1d(input_batch,filters=self.dilation_channels,kernel_size=self.filter_width,dilation_rate=dilation,padding='valid',use_bias=False,name='conv_gate')    
            
            if global_condition_batch is not None:
                conv_filter += tf.layers.conv1d(global_condition_batch,filters=self.dilation_channels,kernel_size=1,padding="same",use_bias=self.use_biases,name="gc_filter")
                conv_gate += tf.layers.conv1d(global_condition_batch,filters=self.dilation_channels,kernel_size=1,padding="same",use_bias=self.use_biases,name="gc_gate")
    
    
            out = tf.tanh(conv_filter) * tf.sigmoid(conv_gate)
    
            # The 1x1 conv to produce the residual output  == FC
            transformed = tf.layers.conv1d(out,filters=self.residual_channels,kernel_size=1,padding="same",use_bias=self.use_biases,name="dense")
    
            # The 1x1 conv to produce the skip output
            # tf.shape(out)[1] <--- 이 값은 dilation에 따라 점점 작아진다. [105114 -> 105112 -> ... -> 100512 -> 100000(output_width)]

            skip_cut = tf.shape(out)[1] - output_width
            out_skip = tf.slice(out, [0, skip_cut, 0], [-1, -1, self.dilation_channels])
            skip_contribution = tf.layers.conv1d(out_skip,filters=self.skip_channels,kernel_size=1,padding="same",use_bias=self.use_biases,name="skip")
    
            input_cut = tf.shape(input_batch)[1] - tf.shape(transformed)[1]
            input_batch = tf.slice(input_batch, [0, input_cut, 0], [-1, -1, -1])
    
            return skip_contribution, input_batch + transformed   # skip_contribution: 결과값으로 쌓임. input_batch + transformed: 다음 단계의 입력으로 들어감

    def _create_network(self, input_batch, global_condition_batch):  
        '''Construct the WaveNet network.'''
        # global_condition_batch: (batch_size, 1, self.global_condition_channels)  <--- 가운데 1은 크기 1짜리 data FC대신에 conv1d를 적용하기 위해 강제로 넣었다고 봐야 한다.
        
        self._create_queue()
        
        outputs = []
        current_layer = input_batch  # causal cut으로 길이 1이 줄어든 상태
        if self.train_mode==False:
            self.causal_queue = tf.scatter_update(self.causal_queue,tf.range(self.batch_size),tf.concat([self.causal_queue[:,1:,:],input_batch],axis=1) )
            current_layer = self.causal_queue
        

        # Pre-process the input with a regular convolution
        current_layer = self._create_causal_layer(current_layer)  # conv1d를 통과 하면서, (filter_width-1)= 1 만큼 더 줄어 있다.

        # 아래의 output_width는 최대 SAMPLE_SIZE = 100,000까지 이고, 짧은 파일이나 파일의 끝부분이면 더 100,000 안 될 수 있다.
        if self.train_mode==True:
            output_width = tf.shape(input_batch)[1] - self.receptive_field + 1   # 모든 dilated convolution을 통과 한 이후 길이. +1을 하는 이유는 causal cut으로 줄어든 1만큼 다시 더해준다.
        else:
            output_width = 1

        # Add all defined dilation layers.
        with tf.variable_scope('dilated_stack'):
            for layer_index, dilation in enumerate(self.dilations): # [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
                with tf.variable_scope('layer{}'.format(layer_index)):
                    
                    if self.train_mode==False:
                        self.dilation_queue[layer_index] =  tf.scatter_update(self.dilation_queue[layer_index],tf.range(self.batch_size),tf.concat([self.dilation_queue[layer_index][:,1:,:],current_layer],axis=1) )
                        current_layer =  self.dilation_queue[layer_index]
                    
                    output, current_layer = self._create_dilation_layer(current_layer, layer_index, dilation,global_condition_batch, output_width)
                    outputs.append(output)
        with tf.name_scope('postprocessing'):
            # Perform (+) -> ReLU -> 1x1 conv -> ReLU -> 1x1 conv to
            # postprocess the output.
             
                    
            # We skip connections from the outputs of each layer, adding them
            # all up here.
            total = sum(outputs)   # list를 sum하는 것이므로, sum 후, (N,output_width,'skip_channels')
            transformed1 = tf.nn.relu(total)
            conv1 = tf.layers.conv1d(transformed1,filters=self.skip_channels,kernel_size=1,padding="same",use_bias=self.use_biases)
    
            transformed2 = tf.nn.relu(conv1)
            conv2 = tf.layers.conv1d(transformed2,filters=self.quantization_channels,kernel_size=1,padding="same",use_bias=self.use_biases)

        return conv2

    def _one_hot(self, input_batch):
        '''One-hot encodes the waveform amplitudes.

        This allows the definition of the network as a categorical distribution
        over a finite set of possible amplitudes.
        '''
        with tf.name_scope('one_hot_encode'):
            encoded = tf.one_hot(input_batch, depth=self.quantization_channels, dtype=tf.float32)  # (1, ?, 1) --> (1, ?, 1, 256)
            shape = [self.batch_size, -1, self.quantization_channels]
            encoded = tf.reshape(encoded, shape)  # (1, ?, 1, 256) --> (1, ?, 256)
        return encoded

    def _embed_gc(self, global_condition):  # global_condition = global_condition_batch <---- data
        '''Returns embedding for global condition.
        :param global_condition: Either ID of global condition for
               tf.nn.embedding_lookup or actual embedding. The latter is
               experimental.
        :return: Embedding or None
        '''
        
        # self.global_condition_cardinality가 None이 아니며, global_condition 은 gc id이면 되고, 그렇지 않으면, global_condition은 embedding vector가 넘어와야 한다.
        embedding = None
        if self.global_condition_cardinality is not None:
            # Only lookup the embedding if the global condition is presented
            # as an integer of mutually-exclusive categories ...
            embedding_table = tf.get_variable('gc_embedding', [self.global_condition_cardinality, self.global_condition_channels], dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer(uniform=False))   # (2, 32)
            embedding = tf.nn.embedding_lookup(embedding_table,global_condition)
        elif global_condition is not None:
            # ... else the global_condition (if any) is already provided
            # as an embedding.

            # In this case, the number of global_embedding channels must be
            # equal to the the last dimension of the global_condition tensor.
            gc_batch_rank = len(global_condition.get_shape())
            dims_match = (global_condition.get_shape()[gc_batch_rank - 1] == self.global_condition_channels)
            if not dims_match:
                raise ValueError('Shape of global_condition {} does not match global_condition_channels {}.'.format(global_condition.get_shape(),
                                        self.global_condition_channels))
            embedding = global_condition

        if embedding is not None:
            embedding = tf.reshape(embedding,[self.batch_size, 1, self.global_condition_channels])

        return embedding


    def predict_proba_incremental(self, waveform, global_condition=None,name='wavenet'):
        '''Computes the probability distribution of the next sample
        incrementally, based on a single sample and all previously passed
        samples.'''

        if self.scalar_input:
            raise NotImplementedError("Incremental generation does not support scalar input yet.")
        with tf.variable_scope(name):
            encoded = tf.one_hot(waveform, self.quantization_channels)
            encoded = tf.reshape(encoded, [self.batch_size,-1, self.quantization_channels])   # encoded shape=(N,1, 256)
            gc_embedding = self._embed_gc(global_condition)                   # --> shape=(1, 1, 32)
            
            
            raw_output = self._create_network(encoded, gc_embedding)        # 이것이 fast generation algorithm의 핵심  --> (batch_size, 1, 256)
            
            
            out = tf.reshape(raw_output, [self.batch_size, self.quantization_channels])
            proba = tf.cast(tf.nn.softmax(tf.cast(out, tf.float64)), tf.float32)

            return proba

    def loss(self, input_batch, global_condition_batch=None, l2_regularization_strength=None, name='wavenet'):
        '''Creates a WaveNet network and returns the autoencoding loss.

        The variables are all scoped to the given name.
        '''
        with tf.variable_scope(name):
            # We mu-law encode and quantize the input audioform.
            # quantization_channels 크기의 one hot encoding을 적용한 예정. 16bit= 65536개였다면,  quantization_channels로 줄이는 효과가 있다.
            # mu law encoding은 bit를 단순히 줄이는 것보다 advanced된 방식으로 줄인다.
            # input_batch: (batch_size,?,1)  <-- 마지막 1은 channel 1을 의미
            encoded_input = mu_law_encode(input_batch, self.quantization_channels)  # "quantization_channels": 256   ---> (batch_size, ?, 1)

            gc_embedding = self._embed_gc(global_condition_batch) # (self.batch_size, 1, self.global_condition_channels) <--- 가운데 1은 강제로 reshape
            encoded = self._one_hot(encoded_input)      #  (1, ?, quantization_channels=256)
            if self.scalar_input:
                network_input = tf.reshape( tf.cast(input_batch, tf.float32), [self.batch_size, -1, 1])
            else:
                network_input = encoded
                
            # Cut off the last sample of network input to preserve causality.
            network_input_width = tf.shape(network_input)[1] - 1
            network_input = tf.slice(network_input, [0, 0, 0], [-1, network_input_width, self.quantization_channels])

            raw_output = self._create_network(network_input, gc_embedding)  # (batch_size, ?, quantization_channels=256) , (batch_size, 1, self.global_condition_channels)

            with tf.name_scope('loss'):
                # Cut off the samples corresponding to the receptive field
                # for the first predicted sample.
                target_output = tf.slice( tf.reshape( encoded, [self.batch_size, -1, self.quantization_channels]), [0, self.receptive_field, 0],[-1, -1, -1])   # [-1,-1,-1] --> 나머지 모두
                
                # 3 dim array의 loss를 계산학 위해, 2 dim으로 변환한다. batch와 time 부분을 합쳐서 2dim으로 변환
                target_output = tf.reshape(target_output, [-1, self.quantization_channels])
                prediction = tf.reshape(raw_output, [-1, self.quantization_channels])
                loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=target_output)
                reduced_loss = tf.reduce_mean(loss)

                tf.summary.scalar('loss', reduced_loss)

                if l2_regularization_strength is None:
                    return reduced_loss
                else:
                    # L2 regularization for all trainable parameters
                    l2_loss = tf.add_n([tf.nn.l2_loss(v)  for v in tf.trainable_variables() if not('bias' in v.name)])

                    # Add the regularization term to the loss
                    total_loss = (reduced_loss + l2_regularization_strength * l2_loss)

                    tf.summary.scalar('l2_loss', l2_loss)
                    tf.summary.scalar('total_loss', total_loss)

                    return total_loss
