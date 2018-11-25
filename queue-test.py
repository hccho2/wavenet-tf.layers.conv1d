#  coding: utf-8
import tensorflow as tf
import numpy as np


tf.reset_default_graph()
batch_size = 2
filter_width = 3
quantization_channels=5
dilation_channels= 8
dilation = [1,2,4]


generation_mode=False


# make queue
Causal_Queue = tf.Variable(initial_value=tf.zeros(shape=[batch_size,(filter_width),quantization_channels], dtype=tf.float32), name='causal_queue', trainable=False)

Dilation_Queue=[]
for i,d in enumerate(dilation):
    q = tf.Variable(initial_value=tf.zeros(shape=[batch_size,d*(filter_width-1)+1,dilation_channels], dtype=tf.float32), name='dilation_queue'.format(i), trainable=False)
    Dilation_Queue.append(q)

# restore했을 때, Dilation_Queue,Causal_Queue는 0으로 initialization해야 한다.
my_initializer= tf.variables_initializer(Dilation_Queue + [Causal_Queue])



sample = tf.placeholder(tf.float32,shape=[batch_size,None,quantization_channels])
inputs = sample
if generation_mode==True:
    Causal_Queue = tf.scatter_update(Causal_Queue,tf.range(batch_size),tf.concat([Causal_Queue[:,1:,:],sample],axis=1) )
    inputs = Causal_Queue
    


current_layer = tf.layers.conv1d(inputs,filters=dilation_channels,kernel_size=filter_width,use_bias=False,strides=1, padding='valid')

for i,d in enumerate(dilation):
    if generation_mode==True:
        Dilation_Queue[i] =  tf.scatter_update(Dilation_Queue[i],tf.range(batch_size),tf.concat([Dilation_Queue[i][:,1:,:],current_layer],axis=1) )
        current_layer =  Dilation_Queue[i]
    current_layer = tf.layers.conv1d(current_layer,filters=dilation_channels,kernel_size=filter_width,dilation_rate=d,use_bias=False,strides=1, padding='valid')
   
                                                         
              
with tf.Session() as sess:
    
    saver = tf.train.Saver(tf.global_variables())
    if generation_mode==True:
        saver.restore(sess,  tf.train.latest_checkpoint('./'))
        sess.run(my_initializer) # 이 부분이 없으면, checkpoint에서 복원된 값들이 들어 있다.
    
    else:
        sess.run(tf.global_variables_initializer())
    
    if generation_mode==True:
        data = np.random.normal(0,1,batch_size*quantization_channels).reshape(batch_size,1,quantization_channels)
        
    else:
        data = np.random.normal(0,1,batch_size*17*quantization_channels).reshape(batch_size,-1,quantization_channels)

    result,cq,dq = sess.run([current_layer,Causal_Queue,Dilation_Queue],feed_dict={sample:data})
    print(result.shape)
    print('data: ', data)
    print('Causal_Queue: ', cq)
    print('Dilation_Queue: ', dq)

    print('*'*20)
    if generation_mode==True:
        data = np.random.normal(0,1,batch_size*quantization_channels).reshape(batch_size,1,quantization_channels)
        
    else:
        data = np.random.normal(0,1,batch_size*17*quantization_channels).reshape(batch_size,-1,quantization_channels)

    result,cq,dq = sess.run([current_layer,Causal_Queue,Dilation_Queue],feed_dict={sample:data})
    print(result.shape)
    print('data: ', data)
    print('Causal_Queue: ', cq)
    print('Dilation_Queue: ', dq)

    
    saver.save(sess, './/my-wavenet', global_step=3)
        
        
        
        