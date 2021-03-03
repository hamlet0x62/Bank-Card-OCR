import tensorflow as tf
from tf_utils import get_weights, get_bias

CHANNEL_NUMS = 1
CONV_1_SIZE = 3
CONV_1_DEEP = 32
CONV_2_SIZE = 3
CONV_2_DEEP = 64
CONV_3_SIZE = 3
CONV_3_DEEP = 64


CONV_4_SIZE = 3
CONV_4_DEEP = 64
FC_SIZE = 1024


def batch_norm(x, is_train):
    return tf.contrib.layers.batch_norm(inputs=x, decay=0.9,
                                        center=True, scale=True, updates_collections=None,
                                        epsilon=1e-5, is_training=is_train, fused=True,
                                        data_format='NHWC', zero_debias_moving_mean=True,
                                        scope="batch-normalization")


def conv(input_tensor, is_train=True):
    """
    :param input_tensor:  with shape: [batch_size, image_height, image_width]
    :return:
    """
    with tf.variable_scope('layer-conv1'):
        conv1_weights = get_weights([CONV_1_SIZE, CONV_1_SIZE, CHANNEL_NUMS, CONV_1_DEEP])
        conv1_bias = get_bias([CONV_1_DEEP])
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = batch_norm(conv1, is_train)
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
    with tf.variable_scope('layer-conv1-1'):
        conv1_1weights = get_weights([CONV_1_SIZE, CONV_1_SIZE, CONV_1_DEEP, CONV_1_DEEP])
        conv1_1bias = get_bias([CONV_1_DEEP])
        conv1_1 = tf.nn.conv2d(relu1, conv1_1weights, strides=[1, 1, 1, 1], padding="SAME")
        conv1_1 = batch_norm(conv1_1, is_train)
        relu_1_1 = tf.nn.relu(tf.nn.bias_add(conv1_1, conv1_1bias))
    with tf.name_scope('pooling-layer-1'):
        pool1 = tf.nn.max_pool(relu_1_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('layer-conv2'):
        conv2_weights = get_weights([CONV_2_SIZE, CONV_2_SIZE, CONV_1_DEEP, CONV_2_DEEP])
        conv2_bias = get_bias([CONV_2_DEEP])
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = batch_norm(conv2, is_train)
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
    with tf.name_scope('pooling-layer-2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    with tf.variable_scope('layer-conv3'):
        conv3_weights = get_weights([CONV_3_SIZE, CONV_3_SIZE, CONV_2_DEEP, CONV_3_DEEP])
        conv3_bias = get_bias(CONV_3_DEEP)
        conv3 = tf.nn.conv2d(pool2, conv3_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv3 = batch_norm(conv3, is_train)
        relu3 = tf.nn.relu(tf.nn.bias_add(conv3, conv3_bias))
    with tf.variable_scope('layer-conv3-1'):
        conv3_1_weights = get_weights([CONV_3_SIZE, CONV_3_SIZE, CONV_3_DEEP, CONV_3_DEEP])
        conv3_1_bias = get_bias(CONV_3_DEEP)
        conv3_1 = tf.nn.conv2d(relu3, conv3_1_weights, strides=[1, 1, 1, 1], padding='SAME')
        conv3_1 = batch_norm(conv3_1, is_train)
        relu3_1 = tf.nn.relu(tf.nn.bias_add(conv3_1, conv3_1_bias))

    return relu3_1


