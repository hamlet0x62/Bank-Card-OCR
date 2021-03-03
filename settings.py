# coding: utf-8
import os

proj_dir = os.path.dirname(__file__)
#
CARD_NO_TRAINSET_PATH = os.path.join(proj_dir, 'images', 'trainset')
GENERATED_CARD_NO_IMG_DIR = os.path.join(proj_dir, 'images', 'generated')
DIR2SAVE_CARD_NO_RECORDS = os.path.join(proj_dir, 'data', 'tf-records-cardno', 'train')
DIR2SAVE_CARDNO_EVAL_RECORDS = os.path.join(proj_dir, 'data', 'tf-records-cardno', 'eval')

LOG_DIR = os.path.join(proj_dir, 'log-dir')

CARD_NO_RECORDS_PATTERN = os.path.join(DIR2SAVE_CARD_NO_RECORDS, '*.tfrecords')
CARD_NO_EVAL_RECORDS_PATTERN = os.path.join(DIR2SAVE_CARDNO_EVAL_RECORDS, '*.tfrecords')
CARD_NO_CHECKPOINT_SAVE_DIR = os.path.join(proj_dir, 'recognize', 'models', '')
CARD_NO_CHECKPOINT_SAVE_PATH_PREFIX = os.path.join(CARD_NO_CHECKPOINT_SAVE_DIR, 'recognize-model')
CARD_NO_RECOGNIZE_LOG_DIR = os.path.join(proj_dir, 'log-dir', 'recognize', '')


MODEL_PATH = os.path.join(proj_dir, 'models', '')
PROCESSED_CARD_IMAGE_DIR = os.path.join(proj_dir, 'images', 'enlarged')


CARDNO_IMG_HEIGHT = 46
CARDNO_IMG_CHANNELS = 1
CARDNO_IMG_WIDTH = 120


# LSTM RNN settings
LSTM_NEURON_NUMS = 128 # 每个LSTM Cell中的神经元个数
LSTM_STACK_DEEPTH = 2 # 堆叠deep LSTM的LSTM Cell个数
# LSTM输出层的神经元个数
NO_CLASS_NUM = 10 + 1   # one for blank class
# 每个输出神经元对应的数字类别
CLASSES = '0123456789'
NO_NUM = 4


# decoding and ecoding settings
# the no classes  0-9 + space
decode_map = {
    i: cls for i, cls in enumerate(CLASSES)
}
decode_map[NO_CLASS_NUM-1] = ''
encode_map = {
    cls: i for i, cls in enumerate(CLASSES)
}

