# coding: utf-8
import numpy as np

from skimage.util import random_noise
from skimage import transform, color
import math
from functools import wraps
from settings import CARDNO_IMG_WIDTH, CARDNO_IMG_HEIGHT
from itertools import chain
import random


def get_angle_by_value(angle_val):
    return angle_val / 180.0 * math.pi


def preprocess_image(image_array):
    """
    :param image_array: an image array
    :return: 2d np.array
    """
    return color.rgb2gray(image_array)


_min_rotate_angle = -10
_max_rotate_angle = 10
_rotate_range = (_min_rotate_angle, _max_rotate_angle)

_min_v_translate_value = -5
_max_v_translate_value = 5
_v_translate_range = (_min_v_translate_value, _max_v_translate_value)

_min_h_translate_value = -4
_max_h_translate_value = 4
_h_translate_value = (_min_h_translate_value, _max_h_translate_value)

_max_shear_angle_value = 10
_min_shear_angle_value = -10
_shear_angle_range = (_min_shear_angle_value, _max_shear_angle_value)

# 图片的压缩和拉伸都保证高度为46不变
# 以保证能够其卷积特征图能够正确输入LSTM

# 图片拉伸、压缩比率
_min_w_push_ratio = 0.8
_max_w_pull_ratio = 1.5
_reduce_enlarge_interval = 0.1
_reduce_enlarge_ratio_range = (_min_w_push_ratio, _max_w_pull_ratio)

# 缩小图片比率
_min_rescale_ratio = 0.6
_max_rescale_ratio = 0.9
_rescale_range = (_min_rescale_ratio, _max_rescale_ratio)

_process_funcs = []
_func_map = {}

_rnd_noise_types = ['gaussian', 'salt', 's&p', 'pepper']


def get_rnd_container():
    return random_noise(np.zeros([CARDNO_IMG_HEIGHT, CARDNO_IMG_WIDTH]), mode=random.choice(_rnd_noise_types))


def gen_process_util_deco(func_container):
    def process_util_deco(value_range, parse_func=None, interval=1, ignore_value=0):
        return process_util(value_range, parse_func=parse_func,
                            interval=interval, ignore_value=ignore_value, func_container=func_container)

    return process_util_deco


def process_util(value_range, parse_func=None, interval=1,
                 ignore_value=0, is_sequence_rst=False,
                 resize2fixedsize=False, ext_func=None,
                 func_container=_process_funcs):
    """
    为图片处理函数编写的装饰器，抽象了数值循环过程，
    通过收集所有处理函数，并统一处理函数的签名来简化调用
    :param process_func: 图像处理函数
    :param value_range: 传递给处理函数的阈值范围
    :param interval: 阈值的递增单位值
    :param func_container: 装填处理函数的容器
    :param is_sequence_rst: 处理结果是否为序列
    :param parse_func: 解析value_range的函数
    :param ignore_value: 忽略处理该值
    :return:
    """
    start_val, end_val = value_range
    values = [val for val in np.arange(start_val, end_val + interval, interval) if val != ignore_value]
    if parse_func is not None:
        values = [parse_func(val) for val in values]

    def outer(process_func):
        @wraps(process_func)
        def wrapper(image, *args, use_ext_func=True, **kws):
            """

            :param image:
            :param args:
            :param use_ext_func: 是否使用该处理函数的扩展函数， 用于各增强函数之间的互相调用
            :param kws:
            :return:
            """
            processed_rst = [process_func(image, val, *args, **kws) for val in values]
            if ext_func and use_ext_func:
                processed_rst = [*chain(*[ext_func(processed_img) for processed_img in processed_rst], processed_rst)]
            return processed_rst if not resize2fixedsize \
                else [transform.resize(img, [CARDNO_IMG_HEIGHT, CARDNO_IMG_WIDTH]) for img in processed_rst]

        func_container.append(wrapper)
        # 将处理函数放入_func_map以便后续调用
        _func_map[process_func.__name__] = wrapper
        return wrapper

    return outer


def gen_shift_func(shift_value, vertical=True):
    axis = 1 if vertical else 0

    def shift(xy):
        xy[:, axis] += shift_value
        return xy

    return shift


@process_util(_rotate_range, ext_func=lambda image: slim_enlarge_image(image, use_ext_func=False), interval=2)
def rotate_image(image_array, value):
    image = transform.rotate(image_array, value)
    return image


@process_util(_v_translate_range, parse_func=lambda val: (gen_shift_func(val, vertical=True), val), interval=2)
def vertically_translate_image(image_array, parsed_rst):
    warp_func, value = parsed_rst
    warped = transform.warp(image_array, warp_func)
    rnd_container = get_rnd_container()
    if value < 0:
        warped[: -1 * value, :] = rnd_container[: -1 * value, :]
    else:
        warped[-1 * value:, :] = rnd_container[-1 * value:, :]
    return warped


@process_util(_h_translate_value, parse_func=lambda val: (gen_shift_func(val, vertical=False), val), interval=2)
def horizontal_translate_image(image_array, parsed_rst):
    warp_func, value = parsed_rst
    warped = transform.warp(image_array, warp_func)
    rnd_container = get_rnd_container()
    if value < 0:
        warped[:, : -1 * value] = rnd_container[:, : -1 * value]
    else:
        warped[:, -1 * value:] = rnd_container[:, -1 * value:]
    return warped

# 错切图片
@process_util(_shear_angle_range, parse_func=lambda val: transform.AffineTransform(scale=[1, 1],
                                                                                   shear=get_angle_by_value(val)))
def shear_filter_image(image_array, value):
    return transform.warp(image_array, value)


# 放大、缩小图片
@process_util(_rescale_range,
              parse_func=lambda ratio: (
                      int((1 - ratio) / 2 * CARDNO_IMG_HEIGHT), int((1 - ratio) / 2 * CARDNO_IMG_WIDTH), ratio),
              interval=_reduce_enlarge_interval, ext_func=lambda image: slim_enlarge_ext(image))
def slim_enlarge_image(image_array, values):
    x_start, y_start, ratio = values
    rescaled_img = transform.rescale(image_array, ratio)

    return paste2container(rescaled_img, (ratio, ratio))


def slim_enlarge_ext(image):
    v_shift_func = _func_map['vertically_translate_image']
    h_shift_func = _func_map['vertically_translate_image']

    shifted = [*v_shift_func(image, use_ext_func=False)]
    shifted.extend(h_shift_func(image, use_ext_func=False))
    return shifted


# 为保持高度一致， 不对图片进行纵向压缩
# 纵向压缩图片
@process_util(value_range=(0.5, 0.9), interval=0.1, parse_func=lambda ratio: (ratio, 1))
def pull_image(image, ratio):
    rescaled_img = transform.rescale(image, scale=ratio)

    return paste2container(rescaled_img, ratio)


# 横向压缩, 拉伸图片
@process_util(value_range=(0.5, 0.9), interval=0.1,
              parse_func=lambda ratio: (1, ratio))
def push_image(image, ratio):
    rescaled_img = transform.rescale(image, scale=ratio)
    return rescaled_img


def paste2container(image, scale_ratio, container_width=CARDNO_IMG_WIDTH, container_height=CARDNO_IMG_HEIGHT):
    """
    将改变尺寸后的图片， 粘贴到与原尺寸一致的容器上， 多余尺寸的像素用高斯噪声补充
    :param image:
    :param scale_ratio:
    :param container_width:
    :param container_height:
    :return:
    """
    height, width = image.shape
    h_ratio, w_ratio = scale_ratio
    rnd_type = random.choice(_rnd_noise_types)
    container = get_rnd_container()
    x_start, y_start = int(container_height * (1 - h_ratio) / 2), int(container_width * (1 - w_ratio) / 2)

    container[x_start: x_start + height, y_start: y_start + width] = image

    return container


def run_process_funcs(image_array, func_container=_process_funcs):
    rst = []
    for process_func in func_container:
        tmp_rst = process_func(image_array)
        rst.extend(tmp_rst)
    rst.append(image_array)
    return rst


if __name__ == "__main__":
    from skimage import io
    test_img_path = r'C:\Users\han\Desktop\bank-card-recognize\images\trainset\01_0e_0.png'
    test_gray = color.rgb2gray(io.imread(test_img_path))

    rst = run_process_funcs(test_gray)
    print(len(rst))
    for img in rst:
        io.imshow(img)
        io.show()
