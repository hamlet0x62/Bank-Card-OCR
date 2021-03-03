import os
import glob
import tensorflow as tf


def get_img_paths(target_dir, glob_expr):
    img_filenames = glob.glob1(target_dir, glob_expr)

    return [os.path.join(target_dir, filename) for filename in img_filenames]


def generate_filename(dir2save, label, count, ext=None):
    filename = os.path.join(dir2save, '{label}#{count:>4}'.format(label=label, count=count))
    if ext:
        filename = '{filename}.{ext}t'.format(filename=filename, ext=ext)
    return filename


def get_filename(path):
    filename = os.path.split(path)[-1]
    filename_without_ext = filename.split('.')[0]

    return filename_without_ext


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def recursive_parse_xml_to_dict(xml):
    """Recursively parses XML contents to python dict.

    We assume that `object` tags are the only ones that can appear
    multiple times at the same level of a tree.

    Args:
      xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
      Python dictionary holding XML contents.
    """
    if not xml:
        return {xml.tag: xml.text}
    result = {}
    for child in xml:
        child_result = recursive_parse_xml_to_dict(child)
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}
