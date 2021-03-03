import tf_utils
import tensorflow as tf
from recognize import lstm
from recognize import input
import time
from recognize.recognize_settings import CARDNO_IMG_WIDTH, CARD_NO_EVAL_RECORDS_PATTERN,\
    CARDNO_IMG_HEIGHT, MODEL_CKPT_DIR
from skimage import io
EVAL_BATCH_SIZE = 5

ckpt_path = tf.train.latest_checkpoint(MODEL_CKPT_DIR)


def evaluate(num2evaluate=100, interval_sec=60):
    with tf.device('/cpu:0'):
        x2eval = tf.placeholder(tf.float32, shape=[None, CARDNO_IMG_HEIGHT, CARDNO_IMG_WIDTH, 1])
        y_label = tf.sparse_placeholder(tf.int32)
        batch_image, batch_label = input.get_input_item(CARD_NO_EVAL_RECORDS_PATTERN,
                                                           capacity=200 + EVAL_BATCH_SIZE * 3,
                                                           min_after_dequeue=200, batch_size=EVAL_BATCH_SIZE,
                                                           num_threads=1)
        logits, seq_len = lstm.inference(x2eval, is_train=False, batch_size=EVAL_BATCH_SIZE)
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
        decoded_op = lstm.get_decoded_seqs(logits, seq_len)
        loss = lstm.loss(logits, y_label, seq_len)
        evaluated_nums = 0
        correct_preds = 0
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            tf.local_variables_initializer().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess, coord=coord)
            try:
                while True:
                    # ckpt_path = tf.train.latest_checkpoint(CARD_NO_CHECKPOINT_SAVE_PATH)
                    if ckpt_path:
                        print('Loading from ' + ckpt_path)
                        saver.restore(sess, save_path=ckpt_path)
                    else:
                        print('Can\'t find any available check point.')
                        return
                    for i in range(num2evaluate):
                        batch_images_val, batch_labels_val = sess.run([batch_image, batch_label])
                        batch_label_strings = [label.decode('utf-8') for label in batch_labels_val]
                        batch_label_seqs = [tf_utils.encode_label2list(label_string)
                                            for label_string in batch_label_strings]
                        sparse_label_tuples = tf_utils.parse_labels_to_sparse_tuple(batch_label_seqs)
                        decoded_seqs, loss_val = sess.run([decoded_op, loss],
                                                                       feed_dict={x2eval: batch_images_val,
                                                                                  y_label: sparse_label_tuples})
                        valid_seqs = lstm.get_validate_seqs(decoded_seqs, batch_label_seqs)
                        evaluated_nums += EVAL_BATCH_SIZE
                        correct_preds += len(valid_seqs)
                        batch_acc = len(valid_seqs) / EVAL_BATCH_SIZE
                        accuracy = correct_preds / evaluated_nums
                        print('The {epoch}\'s evaluation result is'
                              ' {accuracy}, batch acc is {batch_acc} , batch_loss is {loss_val}'.format(
                            epoch=i, accuracy=accuracy,
                            batch_acc=batch_acc, loss_val=loss_val.mean()))
                        # if batch_acc < 0.9:
                        #     print(valid_seqs, decoded_seqs, batch_label_seqs)
                        #     for decoded_seq, label_seq, image in zip(decoded_seqs, batch_label_seqs,
                        #                                              batch_images_val):
                        #         print(decoded_seq, label_seq)

                    time.sleep(interval_sec)
            finally:
                coord.request_stop()
                coord.join(threads)


def get_processed_img_tensor(img_placeholder):
    gray_g = tf.image.rgb_to_grayscale(img_placeholder)
    converted = tf.image.convert_image_dtype(gray_g, tf.float32)
    resized = tf.image.resize_image_with_pad(converted, target_height=46, target_width=500)
    return tf.expand_dims(resized, 0)


def get_predict_op(img_tensor):
    logits, seq_len = lstm.inference(img_tensor, batch_size=1, is_train=False)
    decoded_seq = lstm.get_decoded_seqs(logits, seq_len)

    return decoded_seq


def predict(images, vis=False):
    x_input = tf.placeholder(tf.uint8, [None, None, 3])
    processed_tensor = get_processed_img_tensor(x_input)
    predict_op = get_predict_op(processed_tensor)

    saver = tf.train.Saver(tf.global_variables())
    assert ckpt_path is not None

    predict_rst = []

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()
        saver.restore(sess, save_path=ckpt_path)
        for image in images:
            if image is None:
                predict_rst.append(None)
                continue
            rst, processed_imgs = sess.run([predict_op, processed_tensor], feed_dict={x_input: image})
            if vis:
                io.imshow(processed_imgs[0].reshape(processed_imgs[0].shape[: 2]))
                io.show()
            predict_rst.append(''.join([str(no) for no in rst[0]]))
    return predict_rst


if __name__ == "__main__":
    evaluate()
