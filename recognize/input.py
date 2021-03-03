# coding: utf-8
import tf_utils
import tensorflow as tf
from recognize.recognize_settings import CARD_NO_RECORDS_PATTERN, CARD_NO_EVAL_RECORDS_PATTERN, \
    CARDNO_IMG_HEIGHT, CARDNO_IMG_WIDTH, CARDNO_IMG_CHANNELS, BATCH_SIZE

_batch_size = BATCH_SIZE
_min_after_dequeue = 5000
_capacity = _min_after_dequeue + 3 * _batch_size


def preprocess_input(filename_queue):
    """
    解析一个TFRecord
    :param filename_queue:
    :return:
    """
    reader = tf.TFRecordReader()
    key, examples = reader.read(filename_queue)
    features = tf.parse_single_example(examples, features={
        'image_raw': tf.FixedLenFeature([], tf.string),
        'label_raw': tf.FixedLenFeature([], tf.string),
        'width': tf.FixedLenFeature([], tf.int64),
    })
    decode = tf.image.decode_png(features['image_raw'])
    converted = tf.image.convert_image_dtype(decode, tf.float32)
    processed = tf.image.resize_image_with_pad(converted, CARDNO_IMG_HEIGHT, CARDNO_IMG_WIDTH)
    processed.set_shape([CARDNO_IMG_HEIGHT, CARDNO_IMG_WIDTH, CARDNO_IMG_CHANNELS])

    return processed, features['label_raw']


def get_input_item(filename_patterns=None, capacity=_capacity, batch_size=_batch_size,
                   min_after_dequeue=_min_after_dequeue, num_threads=4):
    """
    获取一个具有随机打乱功能的数据输入队列
    :param filename_patterns:
    :param capacity:
    :param batch_size:
    :param min_after_dequeue:
    :param num_threads:
    :return:
    """
    if filename_patterns is None:
        filename_patterns = CARD_NO_RECORDS_PATTERN
    files = tf.train.match_filenames_once(filename_patterns)
    filename_queue = tf.train.string_input_producer(files, shuffle=True)
    image, label = preprocess_input(filename_queue)
    image_batch, label_batch = tf.train.shuffle_batch((image, label),
                                                      capacity=capacity, batch_size=batch_size,
                                                      num_threads=num_threads,
                                                      min_after_dequeue=min_after_dequeue)

    return image_batch, label_batch


if __name__ == "__main__":
    with tf.Session() as sess:
        images, labels = get_input_item(filename_patterns=CARD_NO_EVAL_RECORDS_PATTERN)
        coord = tf.train.Coordinator()
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        try:
            labels_val, images_val = sess.run([labels, images])

            label_codes = [tf_utils.encode_label2list(label.decode('utf-8')) for label in labels_val]
            sparse_tuple = tf_utils.parse_labels_to_sparse_tuple(label_codes)

            from skimage import io

            for label, image in zip(label_codes, images_val):
                print(label)
                print(image.max())
                io.imshow(image.reshape(image.shape[:2]))
                print(image.shape)
                io.show()
        finally:
            coord.request_stop()
            coord.join(threads)
