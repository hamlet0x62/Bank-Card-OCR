import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from recognize.recognize_settings import LSTM_NEURON_NUMS,\
    LSTM_STACK_DEEPTH, BATCH_SIZE, DEFAULT_LEARNING_RATE, NO_CLASS_NUM

import recognize.cnn_model as model


def get_weights(shape, regularizer=None):
    weight = tf.get_variable('weight', shape, initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weight))
    return weight


def get_bias(shape):
    return tf.get_variable('bias', shape, initializer=tf.constant_initializer(value=.0), dtype=tf.float32)


def inference(input_tensor, batch_size=BATCH_SIZE, regularizer=None, is_train=True, lstm_keep_prob=0.5):
    """
    返回 lstm 模型的前向推导结果
    :param input_tensor: 输入张量， [None, IMAGE_HEIGHT, IMAGE_WIDTH, CHANNEL_NUM]
    :param batch_size:
    :param regularizer: l2正则优化器
    :param is_train: 标识是否出于训练阶段
    :param lstm_keep_prob: # lstm输出的dropout层保留神经元的概率
    :return: output tensor of inference , squence_lengths for each lstm output
    """
    with tf.variable_scope("cnn"):
        batch_feature_image = model.conv(input_tensor, is_train=is_train)
    _, feature_height, feature_width, channel_nums = batch_feature_image.get_shape().as_list()
    seq_len = tf.fill([batch_size], feature_width)
    batch_feature_image = tf.transpose(batch_feature_image, [0, 2, 1, 3])
    batch_feature_image = tf.reshape(batch_feature_image, [-1, feature_width, feature_height * channel_nums])
    # print(batch_feature_image.get_shape().as_list())
    with tf.variable_scope("lstm"):
        cells = [rnn.LSTMCell(LSTM_NEURON_NUMS) for _ in range(LSTM_STACK_DEEPTH)]
        for i in range(len(cells)):
            if is_train:
                cells[i] = rnn.DropoutWrapper(cell=cells[i], output_keep_prob=lstm_keep_prob)
            # print(cells[i].state_size)

        lstm_stack = rnn.MultiRNNCell(cells=cells)
        initial_state = lstm_stack.zero_state(batch_size, tf.float32)
        outputs, h_state = tf.nn.dynamic_rnn(lstm_stack, batch_feature_image,
                                             sequence_length=seq_len,
                                             initial_state=initial_state)
        reshaped_outputs = tf.reshape(outputs, [-1, LSTM_NEURON_NUMS])

        weights = get_weights([LSTM_NEURON_NUMS, NO_CLASS_NUM], regularizer)
        bias = get_bias([NO_CLASS_NUM])

        logits = tf.matmul(reshaped_outputs, weights) + bias
        reshaped_logits = tf.reshape(logits, shape=[batch_size, -1, NO_CLASS_NUM])
    return reshaped_logits, seq_len


def loss(logits, sparse_labels, seq_len):
    """
    根据label和前向推导层结果， 计算loss
    :param logits: LSTM的Logits层输出
    :param sparse_labels: tf.SparseTensor 类型的 batch labels
    :param seq_length: 每个label的实际长度
    :return:
    """
    ctc_loss = tf.nn.ctc_loss(sparse_labels, logits, seq_len, time_major=False)
    return ctc_loss


def train(loss, global_step, learning_rate=DEFAULT_LEARNING_RATE):
    return tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)


def get_decoded_seqs(logits, seq_length):
    """
    get decoded rst by tf.nn.ctc_beam_search_decoder
    :param logits:
    :param sparse_labels:
    :param seq_length:
    :return:
    """
    assert seq_length is not None, "variable _seq_len should not be None"

    time_major_logits = tf.transpose(logits, [1, 0, 2])
    # print(time_major_logits.get_shape().as_list())

    decoded, log_probs = tf.nn.ctc_beam_search_decoder(time_major_logits,
                                                       sequence_length=seq_length,
                                                       merge_repeated=False)

    return tf.sparse_tensor_to_dense(decoded[0], default_value=-1)


def get_validate_seqs(decoded_seqs, label_seqs, ignored_value=-1):
    """
    除去编码为-1的结果, 获取完整的识别序列
    :param decoded_seqs:
    :param label_seqs:
    :param ignored_value:
    :return:
    """
    validated_seq = []
    for i, label_seq in enumerate(label_seqs):
        decoded_seq = [code for code in decoded_seqs[i] if code != ignored_value]
        if decoded_seq == label_seq:
            validated_seq.append(decoded_seq)
    return validated_seq
