from detection.utils.preprocess import gen_shift_func
from skimage import io, transform, util
import cv2
import numpy as np

import tensorflow as tf
import os
import glob
import uuid

from lxml import etree

from object_detection.utils import dataset_util

from multiprocessing import Process

_accept_file_ext_glob = ['*.jp*g', '*.png']
_flags = tf.app.flags
_flags.DEFINE_string('output_dir', '', 'Output directory for saving records')
flags = _flags.FLAGS
_join_path = os.path.join


class TrainImageRecord:
    __slots__ = ["image", "bbox"]

    def __init__(self, image, bbox):
        self.image = image
        self.bbox = [[str(int(it)) for it in box] for box in bbox]
        print(bbox)

    def save(self, output_dir):
        filename = str(uuid.uuid4().hex)[-12:]
        filepath = "{}.png".format(filename)
        io.imsave(_join_path(output_dir, filepath), self.image)
        with open(_join_path(output_dir, "label-{}.txt".format(filename)), 'wb') as f:
            f.writelines([','.join(box).encode() for box in self.bbox])


def read_etree_from_path(path):
    with open(path, 'r') as f:
        xml_string = f.read()
        return etree.fromstring(xml_string)


def keep_aspect_ratio_resize(image, min_size=480.0, max_size=640.0):
    min_shape = min(image.shape[: 2])
    max_shape = max(image.shape[: 2])

    resize_ratio = min_size / min_shape
    if max_shape * resize_ratio > max_size:
        resize_ratio = max_size / max_shape
    height, width = image.shape[: 2]
    resized_h, resized_w = height * resize_ratio, width * resize_ratio
    resized_h, resized_w = [shape if shape % 16 == 0 else (shape//16 + 1) * 16 for shape in [resized_h, resized_w]]

    return cv2.resize(image, [resized_w, resized_h], interpolation=cv2.INTER_LINEAR)


def gen_more_records(data, max_num_each_axis=30):
    """

    :param data:
    :return: list of  tf.train.Example objects
    """
    image = io.imread(data['path'])
    resized_image = keep_aspect_ratio_resize(image)

    width = int(data['size']['width'])
    height = int(data['size']['height'])
    xmins = [int(obj['bndbox']['xmin']) / width for obj in data['object']]
    ymins = [int(obj['bndbox']['ymin']) / height for obj in data['object']]
    xmaxs = [int(obj['bndbox']['xmax']) / width for obj in data['object']]
    ymaxs = [int(obj['bndbox']['ymax']) / height for obj in data['object']]
    resized_h, resized_w = resized_image.shape[: 2]
    # print(resized_h, resized_w)
    rst = []
    rst.append(TrainImageRecord(resized_image,
                                [(xmin_ * resized_w, ymin_ * resized_h, xmax_ * resized_w, ymax_ * resized_h)
                                 for xmin_, ymin_, xmax_, ymax_ in zip(xmins, ymins, xmaxs, ymaxs)]
                                ))

    if len(data['object']) > 1:
        return rst
    xmin, ymin, xmax, ymax = resized_w * xmins[0], resized_h * ymins[0], resized_w * xmaxs[0], resized_h * ymaxs[0]
    xmin, ymin, xmax, ymax = [int(it) for it in [xmin, ymin, xmax, ymax]]
    left_padding = xmin
    right_padding = resized_w - xmax
    up_padding = ymin
    bottom_padding = resized_h - ymax
    # print(left_padding, right_padding, up_padding, bottom_padding)

    noised_template = util.random_noise(np.ones(shape=[resized_h, resized_w, 3]))  # generate noised image template

    def warp_image_with_val(val_range, is_vertical=True, ignored_val=0):
        warp_rst = []
        for shift_val in val_range:
            if shift_val != ignored_val:
                shift_func = gen_shift_func(shift_val, vertical=is_vertical)
                shift_xmin = xmin if is_vertical else xmin - shift_val
                shift_xmax = xmax if is_vertical else xmax - shift_val
                shift_ymin = ymin - shift_val if is_vertical else ymin
                shift_ymax = ymax - shift_val if is_vertical else ymax
                # print(shift_xmin, shift_ymin, shift_xmax, shift_ymax)
                # print(width, height)
                if shift_xmin < 0 or shift_ymin < 0:
                    break
                elif shift_xmax > resized_w or shift_ymax > resized_h:
                    continue
                # get rid of IndexOutOfRange error
                shift_xmax = resized_w - 1 if shift_xmax == resized_w else shift_xmax
                shift_ymax = resized_h - 1 if shift_ymax == resized_h else shift_ymax
                # print(shift_val)

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

                # # test bounding box coordinates
                # cv2.rectangle(shifted_image, (shift_xmin, shift_ymin), (shift_xmax, shift_ymax),
                #               color=(0, 1.0, 0), thickness=10)
                # io.imshow(shifted_image)
                # io.show()
                warp_rst.append(TrainImageRecord(shifted_image,
                                                 bbox=[(shift_xmin, shift_ymin, shift_xmax, shift_ymax)],
                                                 ))
        print("{}: warp_rst length: {} with val_range length: {}".format(os.getpid(), len(warp_rst), len(val_range)))
        return warp_rst

    h_shift_interval = 1
    v_shift_interval = 1
    while right_padding // h_shift_interval + left_padding // h_shift_interval > max_num_each_axis - 1:
        h_shift_interval += 1

    while bottom_padding // v_shift_interval + up_padding // v_shift_interval > max_num_each_axis - 1:
        v_shift_interval += 1
    h_val_range = np.concatenate((np.arange(-1 * right_padding, 0, h_shift_interval),
                                  np.arange(0, left_padding, h_shift_interval)
                                  ))

    v_val_range = np.concatenate((np.arange(-1 * bottom_padding, 0, v_shift_interval),
                                  np.arange(0, up_padding, v_shift_interval)
                                  ))
    # print(h_val_range, v_val_range)
    rst.extend(warp_image_with_val(h_val_range,
                                   is_vertical=False))
    rst.extend(warp_image_with_val(v_val_range,
                                   is_vertical=True))

    # encoded_img = tf.image.encode_png(image.reshape(height, width, 1)).eval()
    # queue.append(
    #     create_tf_example(encoded_img, filename, height, width, (xmin, ymin, xmax, ymax)).SerializeToString())
    print("{}: Generated {} train records.".format(os.getpid(), len(rst)))

    return rst


class PipelineProcess(Process):
    def __init__(self, dataset, trainset_dir, evalset_dir,
                 *args, left2eval=1, **kws):
        self.dataset = dataset
        self.evalset_dir = evalset_dir
        self.left2eval_per_image = left2eval
        self.trainset_dir = trainset_dir

        super(PipelineProcess, self).__init__(*args, **kws)

    def run(self):
        count = 0
        left2eval_count = 0
        for data in self.dataset:
            more_examples = gen_more_records(data, max_num_each_axis=5)
            count += len(more_examples)
            example_size = len(more_examples)

            for i, record in enumerate(more_examples):
                if example_size - i - 1 < self.left2eval_per_image:
                    left2eval_count += 1
                    record.save(self.evalset_dir)
                else:
                    record.save(self.trainset_dir)
            del more_examples
        print("Wrote {} train tfrecords, reserved {} for evaluation".format(count - left2eval_count,
                                                                            left2eval_count))


def create_train_records_from_xml_files(label_dirs, xml_filenames_list, output_dir,
                                        shard_num=4):
    etrees = [read_etree_from_path(os.path.join(label_dir, filename))
              for label_dir, filenames in zip(label_dirs, xml_filenames_list) for filename in filenames]
    datas = [dataset_util.recursive_parse_xml_to_dict(xml)['annotation'] for xml in etrees]
    datas = [data for data in datas if data.get('object', None)]


    trainset_dir = output_dir
    evalset_dir = _join_path(output_dir, 'eval')
    print("Detected {} valid xml files".format(len(datas)))
    shard_size = len(datas) // shard_num
    processes = []
    for shard_index in range(shard_num):
        start_i = shard_index * shard_size
        end_i = (shard_index + 1) * shard_size if shard_index < shard_num - 1 else len(datas)
        process = PipelineProcess(datas[start_i: end_i], trainset_dir=trainset_dir, evalset_dir=evalset_dir)
        processes.append(process)
        process.start()

    for process in processes:
        process.join()
        print('Pipeline #{} joined.'.format(process.pid))

    print("done.")


def main():
    from detection.detection_settings import cur_root
    label_dir = _join_path(cur_root, 'data, dataset', 'label')
    output_dir = _join_path(cur_root, 'data', 'tf-records-card-detect')
    xml_paths = glob.glob1(label_dir, "*.xml")
    create_train_records_from_xml_files([label_dir], [xml_paths], output_dir=output_dir)


if __name__ == "__main__":
    main()
