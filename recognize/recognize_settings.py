# coding: utf-8
import os

from settings import proj_dir, LOG_DIR, DIR2SAVE_CARDNO_EVAL_RECORDS, DIR2SAVE_CARD_NO_RECORDS, \
    CARDNO_IMG_CHANNELS, CARDNO_IMG_HEIGHT, CARDNO_IMG_WIDTH, LSTM_NEURON_NUMS, LSTM_STACK_DEEPTH, NO_CLASS_NUM

here = os.path.dirname(__file__)  # 获取当前文件夹路径

MODEL_CKPT_DIR = os.path.join(here, 'models')
MODEL_CKPT_PATH_PREFIX = os.path.join(MODEL_CKPT_DIR, 'recognize-model', '')


CARD_NO_RECORDS_PATTERN = os.path.join(DIR2SAVE_CARD_NO_RECORDS, '*.tfrecords')
CARD_NO_EVAL_RECORDS_PATTERN = os.path.join(DIR2SAVE_CARDNO_EVAL_RECORDS, '*.tfrecords')
CARD_NO_RECOGNIZE_LOG_DIR = os.path.join(LOG_DIR, 'recognize')

# training settings
BATCH_SIZE = 5
DEFAULT_LEARNING_RATE = 0.0001