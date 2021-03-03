import glob
import os
import random
from skimage import io
from prepare import enlarge_dataset_utils as utils
import tf_utils
import tensorflow as tf

from settings import CARDNO_IMG_HEIGHT

flags = tf.app.flags
flags.DEFINE_string("input_image_dir", None, "Directory that obtains input images")
flags.DEFINE_string("eval_output_dir", None, "Output directory for saving enlarged evaluated dataset")
flags.DEFINE_string('train_output_dir', None, "Output directory for saving enlarged training dataset")
_flags = flags.FLAGS

required_flags = ['input_image_dir', 'eval_output_dir', 'train_output_dir']
for flag in required_flags:
    flags.mark_flag_as_required(flag)


def parse_single_label(label_string):
    rst = label_string.rsplit('.')[0][:4]
    assert len(rst) == 4
    return rst


def parse_no_from_path(paths, sep=''):
    return sep.join([path.rsplit('.')[0][:4] for path in paths])


def pipeline(rst, batch_images, batch_names):
    image_placeholder = tf.placeholder(tf.float32, shape=[None, CARDNO_IMG_HEIGHT])
    transposed_image = tf.transpose(image_placeholder, perm=[1, 0])
    expanded = tf.expand_dims(transposed_image, -1)
    rescaled_img = tf.image.convert_image_dtype(expanded, tf.uint8)

    encode_op = tf.image.encode_png(rescaled_img)

    with tf.Session() as sess:
        for image, name in zip(batch_images, batch_names):
            label = parse_single_label(name)
            preprocessed_img = utils.preprocess_image(image)
            generated_items = [(encode_op.eval({image_placeholder: processed_img.transpose()}),
                                label, preprocessed_img.shape[1])
                               for processed_img in utils.run_process_funcs(preprocessed_img)]

            rst.extend(generated_items)


def main(argv=None):
    target_dir = _flags.input_image_dir
    dir2save_trainset = _flags.train_output_dir
    dir2save_evalset = _flags.eval_output_dir
    matched_img_names = glob.glob1(target_dir, '*.png')
    random.shuffle(matched_img_names)
    img_names = matched_img_names
    random.shuffle(img_names)

    images = [io.imread(os.path.join(target_dir, path)) for path in img_names]
    shard_nums = 4
    shard_size = len(images) // shard_nums
    print("{} images totally".format(len(images)))
    print("About {} images per thread with {} processes".format(shard_size, shard_nums))

    processes = []
    from multiprocessing import Process, Manager

    manager = Manager()
    process_rst = manager.list()

    for shard_i in range(shard_nums):
        start_i = shard_i * shard_nums
        end_i = (shard_i+1) * shard_nums if shard_i<shard_nums-1 else len(images)
        processes.append(Process(target=pipeline,
                                 args=(process_rst,
                                       images[start_i: end_i],
                                       img_names[start_i: end_i])))

    [process.start() for process in processes]

    for process in processes:
        process.join()
        print("Pipeline #{} joined.".format(process.pid))

    processed_images = [item[0] for item in process_rst]
    labels = [item[1] for item in process_rst]
    widths = [item[2] for item in process_rst]
    tf_utils.create_tf_records(processed_images, labels,
                               widths, dir2save_eval_records=dir2save_evalset,
                               dir2save_train_records=dir2save_trainset)
    print('done.')


if __name__ == "__main__":
    tf.app.run(main)
