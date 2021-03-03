# coding: utf-8
import tensorflow as tf
from settings import proj_dir
from demo.demo_settings import here
from recognize.eval import predict

from PIL.ImageDraw import ImageDraw
from PIL.Image import Image
import PIL
from skimage import io
import numpy as np
import re
import os

_image_pattern = re.compile(r'.*(jpe?g|png)')
FROZEN_GRAPH = os.path.join(here, 'detection', 'frozen_inference_graph.pb')
test_img_dir = os.path.join(here, 'test_images')
output_dir = os.path.join(here, 'test_result')
output_filepath = os.path.join(output_dir, 'result.txt')

image_path_list = []
for dirpath, dirnames, filenames in os.walk(test_img_dir):
    for filename in filenames:
        if _image_pattern.match(filename):
            image_path_list.append(os.path.join(dirpath, filename))


def save_img(img, filename=None):
    if filename is None:
        filename = 'saved.jpeg'
    with tf.gfile.FastGFile(os.path.join(proj_dir, filename), 'wb') as writer:
        img_s = tf.image.encode_jpeg(img).eval()
        writer.write(img_s)


def enlarge_box(ymin, xmin, ymax, xmax, img_shape, padding):
    ymin = ymin - padding if ymin - padding > 0 else 0
    ymax = ymax + padding if ymax + padding < img_shape[0] else img_shape[0]
    xmin = xmin - padding if xmin - padding > 0 else 0
    xmax = xmax + padding if xmax + padding < img_shape[1] else img_shape[1]

    return ymin, xmin, ymax, xmax


def draw_bound_box_on_image(image, xmin, ymin, xmax, ymax, vis=True):
    """
    :param image:
    :param xmin, ymin, xmax, ymax: 归一化后的边角坐标
    :param vis:
    :return:
    """
    pil_image = PIL.Image.fromarray(image)
    draw = ImageDraw(pil_image)
    xmin *= pil_image.width
    xmax *= pil_image.width
    ymin *= pil_image.height
    ymax *= pil_image.height
    xmin, ymin, xmax, ymax = [int(it) for it in [xmin, ymin, xmax, ymax]]
    draw.line([(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax), (xmin, ymin)], width=4, fill='blue')
    np.copyto(image, np.array(pil_image))


detection_graph = tf.Graph()

with detection_graph.as_default() as dt_graph:
    dt_graph_def = tf.GraphDef()
    with tf.gfile.FastGFile(FROZEN_GRAPH, 'rb') as frozen_graph:
        graph_data = frozen_graph.read()
        dt_graph_def.ParseFromString(graph_data)
        tf.import_graph_def(dt_graph_def)

with detection_graph.as_default():
    with tf.Session() as session:

        ops = tf.get_default_graph().get_operations()
        img_list = []
        for img_path in image_path_list:
            img = io.imread(img_path)
            img_list.append(img)

        output_tensor_keys = [
            'num_detections', 'detection_boxes', 'detection_scores', 'detection_classes'
        ]
        output_tensors = [tf.get_default_graph().get_tensor_by_name('import/' + tensor_key + ':0')
                          for tensor_key in output_tensor_keys]
        input_tensor = tf.get_default_graph().get_tensor_by_name('import/image_tensor:0')
        imgs_with_bbox = []
        bound_boxes = []
        for img in img_list:
            height, width = img.shape[: 2]
            feed_dict = {input_tensor: [img]}

            rst_list = \
                session.run(output_tensors, feed_dict=feed_dict)
            rst_list = [rst[0] for rst in rst_list]
            detection_nums, detection_boxes, detection_scores, detection_classes = rst_list
            detection_nums = int(detection_nums)
            detection_nums = 1 if detection_nums > 1 else detection_nums
            if detection_nums == 0:
                bound_boxes.append(None)
            for box in detection_boxes[: detection_nums]:
                ymin, xmin, ymax, xmax = box
                rescaled_xmin, rescaled_xmax = [int(it * width) if it < 1.0 else int(width - 1) for it in [xmin, xmax]]
                rescaled_ymin, rescaled_ymax = [int(it * height) if it < 1.0 else int(height - 1) for it in
                                                [ymin, ymax]]
                bound_boxes.append(img[rescaled_ymin: rescaled_ymax, rescaled_xmin: rescaled_xmax])
                draw_bound_box_on_image(img, xmin, ymin, xmax, ymax)
            imgs_with_bbox.append(img)
        print(len(imgs_with_bbox))
        image_filename_list = [os.path.split(image_path)[-1] for image_path in image_path_list]
        predict_rst = predict(bound_boxes)
        with open(output_filepath, 'w') as output_writer:
            filenames = [image_filename.rsplit('.', 1)[0] for image_filename in image_filename_list]
            output_rst = list(zip(filenames, predict_rst))
            output_rst.sort(key=lambda it: it[0])
            output_writer.writelines('\r\n'.join(['{}:{}'.format(filename, rst)
                                                  for filename, rst in output_rst]))
        for image_filename, img_2save in zip(image_filename_list, imgs_with_bbox):
            if img_2save is None:
                continue
            io.imsave(os.path.join(output_dir, image_filename), img_2save)
        print('done.')
