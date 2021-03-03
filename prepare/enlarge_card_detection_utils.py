from skimage import transform, io, color
from prepare.enlarge_dataset_utils import gen_shift_func, gen_process_util_deco, run_process_funcs

from collections import deque

import itertools

_process_funcs = []

process_util = gen_process_util_deco(_process_funcs)


def run_funcs_in_composition(image_array, composite_num=None, func_container=_process_funcs):
    rst = []

    if composite_num is None:
        composite_num = len(func_container)
    elif composite_num > len(func_container):
        raise ValueError("Composition nums can't be greater than the size of func_container")
    queue = deque()

    for composed_funcs in itertools.combinations(_process_funcs, composite_num):
        queue.clear()
        queue.append(image_array)
        for func in composed_funcs:
            tmp_rst = []
            for img, *_ in queue:
                tmp_rst.extend(func(img))
            queue.clear()
            queue.extend(tmp_rst)
            rst.extend(tmp_rst)
    return rst


@process_util(value_range=(-50, 50), parse_func=lambda val: (gen_shift_func(val), val), interval=10)
def shift_image(image_array, func_val_items):
    shift_func, shift_val = func_val_items
    return transform.warp(image_array, shift_func), shift_val


@process_util(value_range=(-50, 50), parse_func=lambda val: (gen_shift_func(val, vertical=False), val), interval=10)
def shift_image_horizontally(image_array, func_val_items):

    shift_func, shift_val = func_val_items
    return transform.warp(image_array, shift_func), shift_val


def test(test_img_path):

    test_img = io.imread(test_img_path)
    test_img = transform.resize(test_img, [480, 640])

    rst = run_process_funcs(test_img, func_container=_process_funcs)
    print("Generated {} images".format(len(rst)))

    for processed_rst in rst:
        if len(processed_rst) == 2:
            img, shift_val = processed_rst
            print(shift_val)
        else:
            img = processed_rst
        io.imshow(img)
        io.show()

if __name__ == "__main__":
    test_img_path = r'path/to/test-img-dir/test.jpeg'  # change to real path while testing
    test(test_img_path)