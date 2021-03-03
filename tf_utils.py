# coding: utf-8
import tensorflow as tf
import numpy as np
import os

from settings import DIR2SAVE_CARD_NO_RECORDS, \
    CARDNO_IMG_HEIGHT, CARDNO_IMG_CHANNELS, DIR2SAVE_CARDNO_EVAL_RECORDS, \
    encode_map, decode_map

import threading

dir2save_train_records = DIR2SAVE_CARD_NO_RECORDS
dir2save_eval_records = DIR2SAVE_CARDNO_EVAL_RECORDS
_record_format = 'records-{:>04}-of-{:>04}.tfrecords'


def int64_feature(data):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[data]))


def int64_list_feature(data):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=data))


def bytes_feature(data):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))


def encode_label2list(label):
    rst = []
    for ch in label:
        if ch != '_':
            rst.append(encode_map[ch])
    return rst


def decode2label(codes):
    return [decode_map[code] for code in codes]


def create_tf_example(encoded_image, label, width, height=CARDNO_IMG_HEIGHT, channels=CARDNO_IMG_CHANNELS):

    return tf.train.Example(features=tf.train.Features(feature={
        'image_raw': bytes_feature(encoded_image),
        'label_raw': bytes_feature(bytes(label.encode('utf-8'))),
        'width': int64_feature(width),
        'height': int64_feature(height),
        'channels': int64_feature(channels)
    }))


class CreateTFRecordsThread(threading.Thread):
    """
    用于多线程写TFRecord
    """
    def __init__(self, images, labels, widths, filename, *args, **kwargs):
        super(CreateTFRecordsThread, self).__init__(*args, **kwargs)
        self.images = images
        self.labels = labels
        self.filename = filename
        self.widths = widths

    def run(self):
        with tf.python_io.TFRecordWriter(self.filename) as writer:
            examples = [create_tf_example(image, label, width) for image, label, width in
                        zip(self.images, self.labels, self.widths)]
            for example in examples:
                writer.write(example.SerializeToString())
        print("{} has been done.".format(self.filename))


def create_tf_records(images, labels, widths, shard_num=5):
    """
    将传入的图片和标签转换成TFRecord格式
    :param images:
    :param labels:
    :param widths:
    :param shard_num:
    :return:
    """
    eval_data_divider = int(len(images) * 0.95)
    eval_images, eval_labels, eval_widths = [data[eval_data_divider:] for data in (images, labels, widths)]
    eval_filename = os.path.join(dir2save_eval_records, 'eval.tfrecords')
    train_images, train_labels, train_widths = [data[:eval_data_divider] for data in (images, labels, widths)]
    train_shard_size = len(train_images) // shard_num
    threads = []
    print('Writing {} tf records for evaluation.'.format(len(eval_images)))
    threads.append(CreateTFRecordsThread(eval_images, eval_labels, eval_widths, eval_filename))

    print('Writing {} tf records for training.'.format(len(train_images)))
    for start_i in range(shard_num):
        shard_start_i = start_i * train_shard_size
        shard_end_i = (start_i + 1) * train_shard_size if start_i < shard_num - 1 else len(train_images)
        shard_filename = os.path.join(dir2save_train_records,
                                      _record_format.format(start_i, shard_num))
        threads.append(
            CreateTFRecordsThread(train_images[shard_start_i: shard_end_i],
                                  train_labels[shard_start_i: shard_end_i],
                                  train_widths[shard_start_i: shard_end_i], shard_filename))
    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()
    print('done')


def parse_labels_to_sparse_tuple(labels):
    """
    将label转换成tf.ctc_losss所需要的tf.SparseTensor类型
    :param labels:
    :return:
    """
    indices = []
    values = []

    for i, label_seq in enumerate(labels):
        indices_items = zip([i] * len(label_seq), range(len(label_seq)))
        indices.extend(indices_items)
        values.extend(label_seq)
    indices = np.asarray(indices, np.int64)
    max_seq_len = indices.max(0)[1] + 1
    shape = [len(labels), max_seq_len]

    return indices, values, shape


def parse_label_bytes_to_sparse_tuple(label_bytes):
    """
    将bytes类型的label先转码， 再转换成 tf.SparseTensor
    :param label_bytes:
    :return:
    """
    batch_label_strings = [label.decode('utf-8') for label in label_bytes]
    batch_label_codes = [encode_label2list(label) for label in batch_label_strings]
    batch_label_sparse_tuple = parse_labels_to_sparse_tuple(batch_label_codes)

    return batch_label_sparse_tuple


def get_weights(shape, regularizer=None):
    weight = tf.get_variable('weight', shape, initializer=tf.truncated_normal_initializer(stddev=0.1), dtype=tf.float32)
    if regularizer is not None:
        tf.add_to_collection('losses', regularizer(weight))
    return weight


def get_bias(shape):
    return tf.get_variable('bias', shape, initializer=tf.constant_initializer(value=.0), dtype=tf.float32)