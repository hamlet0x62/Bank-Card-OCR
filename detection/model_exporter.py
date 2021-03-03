from detection.model_nets import model_train as model
from detection.detection_settings import MODEL_EXPORT_PATH, MODEL_CKPT_PATH as ckpt_path
import detection.utils.resizer as resizer
import tensorflow as tf
from tensorflow.saved_model.signature_constants import DEFAULT_SERVING_SIGNATURE_DEF_KEY

from absl import flags
import os

flags.DEFINE_string('export_path', MODEL_EXPORT_PATH, 'Directory path to export detection model')
flags.DEFINE_string('version', '1', 'The version of exported detection model')

_flags = flags.FLAGS


def main(argv=None):
    encoded_img = tf.placeholder(tf.string, shape=[])
    x = tf.identity(encoded_img, name='x')

    decoded_img = tf.image.decode_image(x, channels=3)
    decoded_img.set_shape([None, None, 3])
    # convert rgb format into gbr format
    decoded_img = decoded_img[:, :, ::-1]

    shape = tf.shape(decoded_img)
    im_height, im_width, im_channels = [shape[i] for i in range(3)]
    expanded = tf.expand_dims(decoded_img, axis=0)
    resize_im = resizer.keep_aspect_ratio_resize(expanded, im_height, im_width)
    resized_shape = tf.shape(resize_im)
    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    bbox_pred, _, cls_prob = model.model(resize_im)
    sess = tf.InteractiveSession()
    variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
    saver = tf.train.Saver(variable_averages.variables_to_restore())
    ckpt = tf.train.latest_checkpoint(ckpt_path)
    if ckpt is None:
        raise ValueError('Cannot find Latest checkpoint in {}'.format(ckpt_path))
    print('Restoring from {}'.format(ckpt))
    tf.global_variables_initializer().run()
    saver.restore(sess, ckpt)

    builder = tf.saved_model.builder.SavedModelBuilder(export_dir=os.path.join(_flags.export_path, _flags.version))

    x_tensor_info = tf.saved_model.utils.build_tensor_info(x)
    bbox_pred_tensor_info, cls_prob_tensor_info = [tf.saved_model.utils.build_tensor_info(tensor)
                                                   for tensor in [bbox_pred, cls_prob]]
    im_shape_info = tf.saved_model.utils.build_tensor_info(resized_shape)
    predict_signature = tf.saved_model.signature_def_utils.build_signature_def(
        inputs={'image': x_tensor_info},
        outputs={'bbox_pred': bbox_pred_tensor_info,
                 'cls_prob': cls_prob_tensor_info,
                 'im_shape': im_shape_info},
        method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
    )

    builder.add_meta_graph_and_variables(sess,
                                         tags=[tf.saved_model.tag_constants.SERVING],
                                         signature_def_map={
                                             DEFAULT_SERVING_SIGNATURE_DEF_KEY: predict_signature
                                         }
                                         )
    builder.save()
    print('done.')


if __name__ == "__main__":
    tf.app.run()

