#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from tensorflow.contrib.slim.nets import resnet_v2, resnet_utils
import tensorflow as tf
from tensorflow.contrib import layers as layers_lib
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.layers.python.layers import utils
from tensorflow.contrib import slim
from tensorflow.nn import ctc_loss, conv2d
import numpy as np
resnet_v2_block = resnet_v2.resnet_v2_block
resnet_v2 = resnet_v2.resnet_v2

import random

# In[3]:


BATCH_SIZE = 4
RESNET_STRIDE = 16
MAX_LABEL_LENGTH = 4


# In[4]:


def resnet_v2_26_base(inputs,
                 num_classes=None,
                 is_training=True, # True - due to update batchnorm layers
                 global_pool=False,
                 output_stride=1, # effective stride
                 reuse=None,
                 include_root_block=False, #first conv layer. Removed due to max pool supression. We need large receprive field
                 scope='resnet_v2_26'):

    """
    Tensorflow resnet_v2 use only bottleneck blocks (consist of 3 layers).
    Thus, this resnet layer model consist of 26 layers.
    I put stride = 2 on each block due to increase receptive field.

    """
    blocks = [
      resnet_v2_block('block1', base_depth=64, num_units=2, stride=2),
      resnet_v2_block('block2', base_depth=128, num_units=2, stride=2),
      resnet_v2_block('block3', base_depth=256, num_units=2, stride=2),
      resnet_v2_block('block4', base_depth=512, num_units=2, stride=2),
    ]
    return resnet_v2(
      inputs,
      blocks,
      num_classes,
      is_training,
      global_pool,
      output_stride,
      include_root_block,
      reuse=reuse,
      scope=scope)

def make_ocr_net(inputs, num_classes, is_training=False):
    '''
    Creates neural network graph.
    Image width halved and it's define timestamps width (feature sequence length)
    No activation after output (no softmax), due to it's presence at ctc_loss() and beam_search().
    After resnet head features are resized to be [batch,1,width,channel], and after that goes 1x1 conv
    to make anology of dense connaction for each timestamp.

    input: batch of images
    output: tensor of size [batch, time_stamps_width, num_classes]
    '''
    with tf.variable_scope('resnet_base', values=[inputs]) as sc:
        with slim.arg_scope([slim.conv2d],
                              activation_fn=None, normalizer_fn=None):
            net = resnet_utils.conv2d_same(inputs, 64, 7, stride=2, scope='conv1') #root conv for resnet
            net = resnet_v2_26_base(net, output_stride=RESNET_STRIDE, is_training = is_training)[0] # ouput is a tuple of last tensor and all tensors
    with tf.variable_scope('class_head', values=[net]) as sc:
        net = tf.transpose(net, [0,3,1,2]) # next 4 lines due to column to channel reshape. [batch,c,h,w]
        _,c,h,_ = net.get_shape() # depth of input to conv op tensor should be static (defined)
        shape = tf.shape(net)
        net = tf.reshape(net, [shape[0], c*h, 1, shape[3]])
        net = tf.transpose(net,[0,2,3,1]) # back to [batch,h,w,c] = [batch,1,w,features*h]
        net = layers_lib.conv2d(net, num_classes, [1, 1], activation_fn=None) #CTC got softmax [batch,1,w,num_classes]
        net = tf.squeeze(net,1) #[batch,w,num_classes]
        return net

def ctc_loss_layer(sequence_labels, logits, sequence_length):
    """
    Build CTC Loss layer for training
    sequence_length is a list of siquences lengths, len(sequence_length) = batch_size.
    In our case sequences can not be different size due to it origin of images batch,
    which should be of equal size (e.g. padded)
    """
    loss = tf.nn.ctc_loss( sequence_labels,
                           logits,
                           sequence_length,
                           time_major=False,  # [batch_size, max_time, num_classes] for logits
                           ignore_longer_outputs_than_inputs=True )
    total_loss = tf.reduce_mean( loss )
    return total_loss

def get_training(sequence_labels, net_logits, sequence_length,
                   learning_rate=1e-4, decay_steps=2**16, decay_rate=0.9, decay_staircase=False,
                   momentum=0.9):
    """
    Set up training ops
    https://github.com/weinman/cnn_lstm_ctc_ocr/blob/master/src/model_fn.py
    """
    with tf.name_scope( "train" ):
        net_logits_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        loss = ctc_loss_layer(sequence_labels, net_logits, sequence_length)
        # Update batch norm stats [http://stackoverflow.com/questions/43234667]
        extra_update_ops = tf.get_collection( tf.GraphKeys.UPDATE_OPS )
        with tf.control_dependencies( extra_update_ops ):
            # Calculate the learning rate given the parameters
            learning_rate_tensor = tf.train.exponential_decay(
                learning_rate,
                tf.train.get_global_step(),
                decay_steps,
                decay_rate,
                staircase=decay_staircase,
                name='learning_rate' )
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate_tensor,
                beta1=momentum )
            train_op = tf.contrib.layers.optimize_loss(
                loss=loss,
                global_step=tf.train.get_global_step(),
                learning_rate=learning_rate_tensor,
                optimizer=optimizer,
                variables=net_logits_vars)
            tf.summary.scalar('learning_rate', learning_rate_tensor )
    return train_op, loss, learning_rate_tensor

def get_prediction(output_net, seq_len, merge_repeated=False):
    '''
    predict by using beam search
    input: output_net - logits (without softmax) of net
           seq_len - length of predicted sequence
    '''
    net = tf.transpose(output_net, [1, 0, 2]) #transpose to [time, batch, logits]
    decoded, prob = tf.nn.ctc_beam_search_decoder(net, seq_len, merge_repeated=merge_repeated)
    return decoded, prob


# In[5]:


class OCRModel(object):
    def __init__(self, image_height, num_classes, input_image_batch, sequence_labels, is_training=True, learning_rate=1e-4,
                decay_steps=2**16, decay_rate=0.9, decay_staircase=False, momentum=0.9):
        self.image_height = image_height
        self.num_classes = num_classes
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.decay_steps = decay_steps
        self.decay_rate = decay_rate
        self.decay_staircase = decay_staircase
        self.momentum = momentum

        self.build(input_image_batch, sequence_labels)

    def build(self, input_image_batch, sequence_labels):

        self.input_image_batch = input_image_batch
        self.sequence_labels = sequence_labels
        self.feature_seq_length = tf.fill([tf.shape(self.input_image_batch)[0]], tf.shape(self.input_image_batch)[2]//(2*RESNET_STRIDE)) #as we know effective stride

        net = make_ocr_net(self.input_image_batch, self.num_classes, is_training=self.is_training)
        self.train_op, self.loss, self.learning_rate_tensor = get_training(self.sequence_labels, net, self.feature_seq_length,
                                                self.learning_rate, self.decay_steps,
                                                self.decay_rate, self.decay_staircase, self.momentum)
        self.prediction = get_prediction(net, self.feature_seq_length, merge_repeated=False) # tuple(decoded, prob). decoded - list of top paths. I use top1
        self.lev_dist = tf.reduce_mean(tf.edit_distance(tf.cast(self.prediction[0][0], tf.int32), self.sequence_labels))
        tf.summary.scalar('CTC loss', self.loss)
        tf.summary.scalar('Levenshtein distance', self.lev_dist)
        tf.summary.scalar('Learning rate', self.learning_rate_tensor)
        self.merged_summary = tf.summary.merge_all()

from data_gen import data_gen
from tensorflow import data


# In[8]:



import json

all_chars = '0123456789'
num_classes = len(all_chars)
with open('second_filtration.json', 'r') as f:
    data = json.load(f)
MAX_DIM = 1024

char_to_indx = dict(zip(all_chars,range(len(all_chars))))
def string_to_label(string):
    label = [char_to_indx[s]+1 for s in string.decode('utf-8')] # +1 because of sparce tensor in string label zero mean nothing
    return label
def resize_to_max(image, max_dim):
    h, w = image.shape[:-1]
    scale_factor = max_dim/max([h,w])
    output_image = rescale(image.copy(), scale = scale_factor, order = 3)
    return output_image
def scale(image):
    scaled_image = image.copy().astype(np.float)
    scaled_image /= 127.5
    scaled_image -= 1
    return scaled_image
def unscale(image):
    unscaled_image = image.copy() + 1
    unscaled_image *= 127.5
    unscaled_image = unscaled_image.astype(np.uint8)
    return unscaled_image
def pad_to_square(image):
    if image.shape[0] == image.shape[1]:
        return image
    max_dim = max(image.shape[:-1])
    ind_max_dim = image.shape[:-1].index(max_dim)
    min_dim = min(image.shape[:-1])
    ind_min_dim = image.shape[:-1].index(min_dim)
    pad = [[],[],[0,0]]
    pad[ind_max_dim]=[0,0]
    pad[ind_min_dim]=[(max_dim-min_dim)//2, (max_dim-min_dim)//2 + (max_dim-min_dim)%2]
    image_pad = np.pad(image.copy(), pad, 'constant')
    return image_pad

#data generation
#generate batch on python generator stage because samples removment got be possible
#def url_label_gen():
#    while True:
#        random.shuffle(data)
#        batch_url = []
#        batch_string = []
#        for num, example in enumerate(data):
#            string = str(example['price_rub'])
#            batch_url.append(image)
#            batch_string.append(string)
#            if len(batch_string) == BATCH_SIZE:
#                yield batch_url, batch_string
#                batch_url = []
#                batch_string = []
#def read_out_image(batch_url, batch_string):
#    batch_image = []
#    batch_label = []
#    for url in batch_url:
#        try:
#            image = io.imread(url)
#        except:
#            print('Cannot read image {}'.format(url))
#            continue
#        if len(image.shape) !=3 or image.shape[-1] != 3:
#            print('image {} has not proper shape {}'.format(url, image.shape))
#            continue
#        image = scale(image)
#        image = resize_to_max(image, max_dim=MAX_DIM)
#        image = pad_to_square(image)
#        batch_image.append(image)
#    for string in batch_string:
#        label = string_to_label(string)
#        batch_label.append(label + [0]*(MAX_LABEL_LENGTH - len(label)))
#        return np.array(batch_image), np.arrat(batch_label)

def url_label_gen():
    while True:
        random.shuffle(data)
        for num, example in enumerate(data):
            string = str(example['price_rub'])
            url = example['url']
            yield (url, string)
from skimage import io
from skimage.transform import rescale
def read_out_image(url, string):
    try:
        image = io.imread(url.decode('utf-8'))
    except Exception as e:
        print(str(e))
        print('Cannot read image {}'.format(url))
        image = None
    if image is not None:
        if len(image.shape) !=3 or image.shape[-1] != 3:
            print('image {} has not proper shape {}'.format(url, image.shape))
            image = None
    if image is None:
       image = 2*np.random.rand(1024,1024,3) - 1
       label = np.zeros(MAX_LABEL_LENGTH) + num_classes+2 #empty sign
       return np.array(image, dtype=np.float32), np.array(label, dtype=np.int32)
    image = scale(image)
    image = resize_to_max(image, max_dim=MAX_DIM)
    image = pad_to_square(image)
    label = string_to_label(string)
    label = label + [0]*(MAX_LABEL_LENGTH - len(label))
    return np.array(image, dtype=np.float32), np.array(label, dtype=np.int32)

def wrapped_complex_calulation(url, string):
    '''
    to make download parallel
    https://stackoverflow.com/questions/47086599/parallelising-tf-data-dataset-from-generator/47089278
    '''
    image_tensor, label_tensor = tf.py_func(func = read_out_image,
                                            inp = (url, string),
                                            Tout = (tf.float32,    # (H,W,3) img
                                                    tf.int32) #labels
                                            )
    image_tensor.set_shape([1024, 1024, 3]) #for the conv layer we should specify channels number
    return image_tensor, label_tensor

# In[9]:

graph = tf.Graph()
with graph.as_default():
    dataset = tf.data.Dataset.from_generator(url_label_gen, output_types= (tf.string, tf.string),
                                              output_shapes = (tf.TensorShape([]),
                                                                           tf.TensorShape([])))
    dataset = dataset.map(wrapped_complex_calulation, num_parallel_calls=4).batch(BATCH_SIZE).prefetch(4)
    data_iter = dataset.make_initializable_iterator()
    features, labels = data_iter.get_next()
    labels = tf.contrib.layers.dense_to_sparse(labels, )
    tf.train.create_global_step()
    model = OCRModel(image_height=1024, num_classes=num_classes+2, input_image_batch=features,
                     sequence_labels=labels, is_training=True)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


# In[ ]:


import time
tic = time.time()
with tf.Session(graph=graph) as sess:
    train_writer = tf.summary.FileWriter('log/train',
                                          sess.graph)
    test_writer = tf.summary.FileWriter('log/test')
    sess.run([init, data_iter.initializer])
    try:
        saver.restore(sess, "log/model.ckpt")
    except:
        print('Cannot restore model')
    for num in range(int(5e6)):
        print(num,end='\r')
        _, ms = sess.run([model.train_op, model.merged_summary])
        train_writer.add_summary(ms, num)
        if num%100 == 0:
            _, ms_ = sess.run([model.train_op, model.merged_summary])
            test_writer.add_summary(ms_, num)
print(time.time()-tic)
