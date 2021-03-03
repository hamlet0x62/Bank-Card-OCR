# coding: utf-8
from recognize.input import get_input_item
import tf_utils
import tensorflow as tf
from recognize import lstm
from recognize.recognize_settings import MODEL_CKPT_DIR, MODEL_CKPT_PATH_PREFIX, BATCH_SIZE, \
    CARDNO_IMG_WIDTH, CARDNO_IMG_HEIGHT, CARD_NO_EVAL_RECORDS_PATTERN, CARD_NO_RECOGNIZE_LOG_DIR

regularize_lambda = 0.0001
learning_rate_base = 0.0001  # 学习率
decay_rate = 0.99  # 衰减率
decay_steps = 10000

train_steps = 30000  # 每次迭代的训练次数
num_epoch = 10  # 迭代次数

image_batch, label_string_batch = get_input_item()
validate_images, validate_labels = get_input_item(filename_patterns=CARD_NO_EVAL_RECORDS_PATTERN, num_threads=1)

x2feed = tf.placeholder(tf.float32, [None, CARDNO_IMG_HEIGHT, CARDNO_IMG_WIDTH, 1])  # one channel
y_labels = tf.sparse_placeholder(tf.int32)

# l2_regularizer = tf.contrib.layers.l2_regularizer(regularize_lambda)
logits, seq_len = lstm.inference(x2feed, batch_size=BATCH_SIZE, regularizer=None)

loss = lstm.loss(logits, y_labels, seq_len=seq_len)

global_step = tf.Variable(0, trainable=False, dtype=tf.int64)
learning_rate = tf.train.exponential_decay(learning_rate_base, global_step,
                                           decay_rate=decay_rate, decay_steps=decay_steps)
# edit_dist = lstm.edit_distance(logits, y_labels, seq_len)
decoded_op = lstm.get_decoded_seqs(logits, seq_len)
train_op = lstm.train(loss, global_step, learning_rate)
init_op = tf.group(tf.local_variables_initializer(), tf.global_variables_initializer())

# summaries
loss_summary = tf.summary.scalar('loss', tf.reduce_mean(loss))
learning_rate_summary = tf.summary.scalar('learning_rate', learning_rate)
image_summary = tf.summary.image('eval_image', x2feed, max_outputs=5)
scalar_summary = tf.summary.merge([loss_summary, learning_rate_summary])
# merged_summary = tf.summary.merge_all()
saver = tf.train.Saver(max_to_keep=100)

newest_ckpt = tf.train.latest_checkpoint(MODEL_CKPT_DIR)
if newest_ckpt:
    from prepare.utils import get_filename

    print(newest_ckpt)
    step_val = int(get_filename(newest_ckpt).rsplit('-', 1)[-1])
    tf.assign(global_step, step_val)

with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter(CARD_NO_RECOGNIZE_LOG_DIR, sess.graph)

    sess.run(init_op)
    if newest_ckpt:
        saver.restore(sess, newest_ckpt)
    coordinator = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coordinator, sess=sess)
    try:
        for epoch in range(num_epoch):
            for train_step in range(1, train_steps + 1):
                label_bytes_batch_val, batch_images_val = sess.run([label_string_batch, image_batch])
                batch_label_sparse_tuple = tf_utils.parse_label_bytes_to_sparse_tuple(label_bytes_batch_val)
                _, loss_val, lr_val, summary_val, global_step_val = sess.run(
                    [train_op, loss, learning_rate, scalar_summary, global_step],
                    feed_dict={y_labels: batch_label_sparse_tuple, x2feed: batch_images_val})
                summary_writer.add_summary(summary_val, global_step=global_step_val)
                if train_step % 100 == 0:
                    print("[{} steps totally] After {} steps training of {} epoch, the loss is {}, with lr {}".format(
                        global_step_val,
                        train_step, epoch,
                        loss_val, lr_val))
                    if train_step % 5000 == 0:
                        validate_images_val, validate_label_bytes = sess.run([validate_images, validate_labels])
                        validate_labels_string = [label_bytes.decode('utf-8') for label_bytes in validate_label_bytes]
                        validate_labels_code = [tf_utils.encode_label2list(label_string) for label_string in
                                                validate_labels_string]
                        validate_lables_tuple = tf_utils.parse_labels_to_sparse_tuple(validate_labels_code)
                        decoded_seq, validate_loss_val, image_summary_val = \
                            sess.run([decoded_op, loss,
                                      image_summary],
                                     feed_dict={
                                         x2feed: validate_images_val,
                                         y_labels: validate_lables_tuple
                                     })
                        summary_writer.add_summary(image_summary_val, global_step=global_step_val)
                        validate_seqs = lstm.get_validate_seqs(decoded_seq, validate_labels_code)
                        validate_acc = len(validate_seqs) / BATCH_SIZE
                        print(validate_seqs, decoded_seq, validate_labels_code)
                        print("The validation acc is {}, validate loss is {}".format(validate_acc,
                                                                                     validate_loss_val))
                        saver.save(sess, save_path=MODEL_CKPT_PATH_PREFIX, global_step=global_step)
    except Exception as e:
        coordinator.request_stop(e)
        raise e
    finally:
        coordinator.request_stop()
        coordinator.join(threads)
