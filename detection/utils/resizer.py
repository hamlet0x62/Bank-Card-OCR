import tensorflow as tf


def _to_float(tensor):
    return tf.to_float(tensor)


def keep_aspect_ratio_resize(im_tensor, im_height, im_width, im_channels=3):
    float_type_h = _to_float(im_height)
    float_type_w = _to_float(im_width)
    min_shape = tf.minimum(float_type_h, float_type_w)
    max_shape = tf.maximum(float_type_h, float_type_w)

    resize_ratio = 480.0 / min_shape
    resized_max_shape = _to_float(max_shape) * resize_ratio

    other_resize_ratio = 640.0 / max_shape
    final_resize_ratio = tf.case([(tf.greater(resized_max_shape, 640.0), lambda: other_resize_ratio)],
                                 default=lambda: resize_ratio)
    resized_h = tf.to_int32(float_type_h * final_resize_ratio)
    resized_w = tf.to_int32(float_type_w * final_resize_ratio)

    h_mod_rst = tf.mod(resized_h, 16)
    target_h = tf.case([(tf.equal(h_mod_rst, 0), lambda: resized_h)], default=lambda: resized_h + 16 - h_mod_rst)
    w_mod_rst = tf.mod(resized_w, 16)
    target_w = tf.case([(tf.equal(w_mod_rst, 0), lambda: resized_w)], default=lambda: resized_w + 16 - w_mod_rst)

    return tf.image.resize_bilinear(im_tensor, [target_h, target_w])
