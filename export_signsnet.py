# zhangshulin
# zhangslwork@yeah.net
# 2018-2-10


import tensorflow  as tf
from tensorflow.python.framework import graph_util
from alexnet import AlexNet


CHECKPOINT_PATH = './save/alexnet_train_vars'

EXPORT_PATH = './signsnet.pd'


def export_signsnet(ckpt, export_path):
    tf.reset_default_graph()
    sess = tf.Session()

    alexnet = AlexNet(image_shape=(68, 68, 3), classes_num=6, freeze_layer_indexes = (1, 2, 3, 4, 5))
    sess.run(tf.global_variables_initializer())
    alexnet.load_imagenet_weights(sess, skip_layer_indexes=(6, 7, 8))

    saver = tf.train.Saver(tf.trainable_variables())
    saver.restore(sess, CHECKPOINT_PATH)

    graph_def = tf.get_default_graph().as_graph_def()
    output_graph_def = graph_util.convert_variables_to_constants(sess, graph_def, ['softmax8/softmax'])

    with tf.gfile.GFile(EXPORT_PATH, 'wb') as f:
        f.write(output_graph_def.SerializeToString())

    print('Export success ! File: {}'.format(EXPORT_PATH))

    sess.close()


if __name__ == '__main__':
    export_signsnet(CHECKPOINT_PATH, EXPORT_PATH)
