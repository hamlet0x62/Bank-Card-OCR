import matplotlib.pyplot as plt
from prepare.enlarge_dataset_utils import gen_shift_func
from settings import proj_dir
from skimage import io, color, transform, util
from PIL import Image
import numpy as np

import tensorflow as tf
import random
import os
import glob
import pickle
import uuid

from lxml import etree

import prepare.utils as dataset_util

from multiprocessing import Process
GENERATE_NUMS_PER_IMG = 10

_accept_file_ext_glob = ['*.jp*g', '*.png']
_generated_img_dir = os.path.join(proj_dir, 'data', 'bank-cards', 'generated_imgs')
_counter_path = os.path.join(proj_dir, 'data', 'bank-cards', 'file_counter.pkl')
_counter = None

flags = tf.app.flags
flags.DEFINE_string("label_dir", None, 'Directory that saved labels')
flags.DEFINE_string('train_output_dir', None, 'Output directory for saving training dataset')
flags.DEFINE_string('eval_output_dir', None, 'Output directory for saving evaluation dataset')
_flags = flags.FLAGS

if not os.path.exists(_counter_path):
    _counter = 0
    with open(_counter_path, 'wb') as f:
        pickle.dump(_counter, f)
else:
    with open(_counter_path, 'rb') as f:
        _counter = pickle.load(f)


def imshow_array(img_array):
    plt.imshow(img_array)
    plt.show()


def keep_aspect_ratio_resize(image, min_size=600.0, max_size=1200.0):
    min_shape = min(image.shape[: 2])
    max_shape = max(image.shape[: 2])

    resize_ratio = min_size / min_shape
    if max_shape * resize_ratio > max_size:
        resize_ratio = max_size / max_shape
    height, width = image.shape[: 2]
    resized_h, resized_w = height * resize_ratio, width * resize_ratio
    return transform.resize(image, output_shape=[int(resized_h), int(resized_w)], mode="symmetric")


def read_etree_from_path(path):
    with open(path, 'r') as f:
        xml_string = f.read()
        return etree.fromstring(xml_string)


def create_tf_example(encoded, unique_filename, height, width, coordinates):
    xmins, ymins, xmaxs, ymaxs = coordinates
    example = tf.train.Example(
        features=tf.train.Features(
            feature={
                'image/height': dataset_util.int64_feature(height),
                'image/width': dataset_util.int64_feature(width),
                'image/source_id': dataset_util.bytes_feature(unique_filename),
                'image/encoded': dataset_util.bytes_feature(encoded),
                'image/format': dataset_util.bytes_feature(b'png'),
                'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
                'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
                'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
                'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
                'image/object/class/text': dataset_util.bytes_list_feature([b'bank-card-no'] * len(xmins)),
                'image/object/class/label': dataset_util.int64_list_feature([1] * len(xmins)),

            }
        )
    )

    return example


def gen_more_tf_examples(data, image_placeholder, encode_op, max_num_each_axis=30):
    """

    :param data:
    :return: list of  tf.train.Example objects
    """
    image = io.imread(data['path'])

    width = int(data['size']['width'])
    height = int(data['size']['height'])
    xmins = [int(obj['bndbox']['xmin']) / width for obj in data['object']]
    ymins = [int(obj['bndbox']['ymin']) / height for obj in data['object']]
    xmaxs = [int(obj['bndbox']['xmax']) / width for obj in data['object']]
    ymaxs = [int(obj['bndbox']['ymax']) / height for obj in data['object']]
    resized_image = keep_aspect_ratio_resize(image)
    resized_h, resized_w =[int(shape) for shape in resized_image.shape[: 2]]
    rst = []
    encoded_image = encode_op.eval({image_placeholder: image})
    example = create_tf_example(encoded_image, uuid.uuid4().bytes, height, width, (xmins, ymins, xmaxs, ymaxs))
    rst.append(example.SerializeToString())

    if len(data['object']) > 1:
        return rst
    xmin, ymin, xmax, ymax = resized_w * xmins[0], resized_h * ymins[0], resized_w * xmaxs[0], resized_h * ymaxs[0]
    xmin, ymin, xmax, ymax = [int(it) for it in [xmin, ymin, xmax, ymax]]
    left_padding = xmin
    right_padding = resized_w - xmax
    up_padding = ymin
    bottom_padding = resized_w - ymax
    width = resized_w
    height =  resized_h

    noised_template = util.random_noise(np.ones(shape=[height, width, 3]))  # generate noised image template

    def warp_image_with_val(val_range, is_vertical=True, ignored_val=0):
        for shift_val in val_range:
            if shift_val != ignored_val:
                shift_func = gen_shift_func(shift_val, vertical=is_vertical)  # 根据位移方向和位移量， 获得位移函数
                shift_xmin = xmin if is_vertical else xmin - shift_val
                shift_xmax = xmax if is_vertical else xmax - shift_val

                shift_ymin = ymin - shift_val if is_vertical else ymin
                shift_ymax = ymax - shift_val if is_vertical else ymax
                # print(shift_xmin, shift_ymin, shift_xmax, shift_ymax)
                # print(width, height)
                if shift_xmin < 0 or shift_ymin < 0:
                    break
                elif shift_xmax > width or shift_ymax > height:
                    continue
                shift_xmax = width - 1 if shift_xmax == width else shift_xmax
                shift_ymax = height - 1 if shift_ymax == height else shift_ymax
                shifted_image = transform.warp(resized_image, shift_func)

                # add gaussian noise to pure-black areas
                if is_vertical:
                    if shift_val > 0:
                        shifted_image[-1 * shift_val:] = noised_template[-1 * shift_val:]
                    else:
                        shifted_image[: -1 * shift_val] = noised_template[: -1 * shift_val]
                else:
                    if shift_val > 0:
                        shifted_image[:, -1 * shift_val:] = noised_template[:, -1 * shift_val:]
                    else:
                        shifted_image[:, :-1 * shift_val] = noised_template[:, :-1 * shift_val]

                # test bounding box coordinates
                # shifted_image[shift_ymin, shift_xmin: shift_xmax] = 1
                # shifted_image[shift_ymax, shift_xmin: shift_xmax] = 1
                # io.imshow(shifted_image)
                # io.show()
                encoded = encode_op.eval(feed_dict={image_placeholder: shifted_image})
                coordinates = [shift_xmin / width], [shift_ymin/ height], [shift_xmax/width], [shift_ymax / height]
                rnd_filename = uuid.uuid4().bytes
                example = create_tf_example(encoded, rnd_filename, height, width, coordinates)
                rst.append(example.SerializeToString())

    h_shift_interval = 1
    v_shift_interval = 1
    while right_padding // h_shift_interval + left_padding // h_shift_interval > max_num_each_axis-1:
        h_shift_interval += 1

    while bottom_padding // v_shift_interval + up_padding // v_shift_interval > max_num_each_axis-1:
        v_shift_interval += 1
    h_val_range = np.concatenate((np.arange(-1 * right_padding, 0, h_shift_interval),
                                  np.arange(0, left_padding, h_shift_interval)
                                  ))

    v_val_range = np.concatenate((np.arange(-1 * bottom_padding, 0, v_shift_interval),
                                  np.arange(0, up_padding, v_shift_interval)
                                  ))
    # print(h_val_range, v_val_range)
    warp_image_with_val(h_val_range,
                        is_vertical=False)
    warp_image_with_val(v_val_range,
                        is_vertical=True)

    # encoded_img = tf.image.encode_png(image.reshape(height, width, 1)).eval()
    # queue.append(
    #     create_tf_example(encoded_img, filename, height, width, (xmin, ymin, xmax, ymax)).SerializeToString())
    print("Generated {} examples.".format(len(rst)))

    return rst


class PipelineProcess(Process):
    def __init__(self, dataset, trainset_filename, evalset_filename,
                 *args, left2eval=1, **kws):
        self.dataset = dataset
        self.eval_filename = evalset_filename
        self.left2eval_per_image = left2eval
        self.train_filename = trainset_filename

        super(PipelineProcess, self).__init__(*args, **kws)

    def run(self):
        count = 0
        left2eval_count = 0
        image2feed = tf.placeholder(tf.float32, [None, None, 3])
        rescaled_image = tf.image.convert_image_dtype(image2feed, tf.uint8)
        encode_op = tf.image.encode_png(rescaled_image)
        with tf.Session() as sess:
            gen_record_writer = tf.python_io.TFRecordWriter
            with gen_record_writer(self.train_filename) as train_writer, \
                    gen_record_writer(self.eval_filename) as eval_writer:
                for data in self.dataset:
                    more_examples = gen_more_tf_examples(data, image2feed, encode_op, max_num_each_axis=5)
                    count += len(more_examples)
                    example_size = len(more_examples)

                    for i, record in enumerate(more_examples):
                        if example_size - i - 1 < self.left2eval_per_image:
                            left2eval_count += 1
                            eval_writer.write(record)
                        else:
                            train_writer.write(record)
                    del more_examples
        print("Wrote {} train tfrecords, reserved {} for evaluation".format(count - left2eval_count,
                                                                            left2eval_count))


def create_tf_examples_from_xml_files(label_dirs, xml_filenames_list,
                                      output_dir, eval_output_dir,
                                      shard_num=4):
    etrees = [read_etree_from_path(os.path.join(label_dir, filename))
              for label_dir, filenames in zip(label_dirs, xml_filenames_list) for filename in filenames]
    datas = [dataset_util.recursive_parse_xml_to_dict(xml)['annotation'] for xml in etrees]
    datas = [data for data in datas if data.get('object', None)]

    trainset_filename = 'train-set-record-{:>03}-of-{:>04}.tfrecords'
    evalset_filename = 'eval-set-record-{:>01}-of-{:>02}.tfrecords'

    print("Detected {} valid xml files".format(len(datas)))
    shard_size = len(datas) // shard_num
    join_path = os.path.join
    processes = []
    for shard_index in range(shard_num):
        start_i = shard_index * shard_size
        end_i = (shard_index + 1) * shard_size if shard_index < shard_num - 1 else len(datas)

        cur_shard_filename = join_path(output_dir, trainset_filename.format(shard_index,
                                                                            shard_num))

        cur_shard_eval_filename = join_path(eval_output_dir, evalset_filename.format(shard_index, shard_num))
        process = PipelineProcess(datas[start_i: end_i], cur_shard_filename,
                                  cur_shard_eval_filename)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
        print('Pipeline #{} joined.'.format(process.pid))

    print("done.")


def main(argv=None):
    label_dir = _flags.label_dir
    output_dir = _flags.output_dir
    eval_output_dir = flags.eval_output_dir
    if any(it is None for it in [label_dir, output_dir]):
        return None
    for required_flag in ['label_dir', 'eval_output_dir', 'train_output_dir']:
        flags.mark_flag_as_required(required_flag)

    xml_paths = glob.glob1(label_dir, "*.xml")
    create_tf_examples_from_xml_files([label_dir], [xml_paths], output_dir=output_dir,
                                      eval_output_dir=eval_output_dir)


if __name__ == "__main__":
    tf.app.run(main)
