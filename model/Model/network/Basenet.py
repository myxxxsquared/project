"""
return a conv net that takes a image as input and return the activations of all layers
"""
import numpy as np
import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer_conv2d as xavier
from . import ResNetSchedule


def cpu_variable_getter(getter, *args, **kwargs):
    with tf.device("/cpu:0"):
        return getter(*args, **kwargs)


class BaseNet(object):
    def __init__(self, configs, log):
        self.input_size = configs.input_size  # tuple  height * width * channels
        self.padding = configs.padding  # str
        self.pooling = configs.pooling
        self.log = log

    @staticmethod
    def minus_mean(image, MeanPixel):
        # this step in carried out in VGG
        return image - MeanPixel

    @staticmethod
    def undo_minus_mean(image, MeanPixel):
        # this step in carried out in VGG
        return image + MeanPixel

    def pre_processing(self, image):
        return self.minus_mean(image, np.reshape((123.68, 116.78, 103.94), (1, 1, 1, 3))) / 256

    def undo_pre_processing(self, image):
        return self.undo_minus_mean(image, np.reshape((123.68, 116.78, 103.94), (1, 1, 1, 3))) * 256

    @staticmethod
    def ConvLayer(inputs, filters=None, bias=None, shape=None, strides=(1, 1, 1, 1), padding="SAME", name="Conv", trainable=True, initializer=xavier):
        """
        :param inputs:
        :param filters:
        :param bias:
        :param shape: (k1,k2,input_channel,output_channel)
        :param padding:"SAME"(zero),"VALID"(no)
        :param name:
        :param trainable:
        :return:
        """
        if filters:
            conv = tf.nn.conv2d(
                inputs,
                tf.get_variable(name, initializer=filters, dtype=tf.float32, trainable=trainable),
                strides=strides,
                padding=padding,
                name=name)
        elif shape:
            conv = tf.nn.conv2d(
                inputs,
                tf.get_variable(name, dtype=tf.float32, shape=shape, initializer=initializer(), trainable=True),
                strides=strides,
                padding=padding,
                name=name)
        else:
            raise ValueError('At least one of the following should be passed: shape, filters')
        if bias:
            return tf.nn.bias_add(conv, bias)
        else:
            return tf.nn.bias_add(conv, tf.get_variable(name + '_bias', dtype=tf.float32, shape=(shape[-1]), initializer=initializer()))

    @staticmethod
    def Pooling(inputs, pooling, padding="SAME", name="Pooling", strides=(1, 2, 2, 1), ksize=(1, 2, 2, 1)):
        pools = {"max": tf.nn.max_pool, "avg": tf.nn.avg_pool}
        return pools[pooling](
            inputs,
            ksize=ksize,
            strides=strides,
            padding=padding,
            name=name)

    @staticmethod
    def upsampling(inputs, upsampling='DeConv', **kwargs):
        """
        :param inputs:
        :param upsampling: UpSam algorithm, 'BiLi'or'DeConv' or 'unpooling'(?)
        :param name: str
        :param kwargs: dict{}
        :return:
        conv2d_transpose(
            inputs,
            filters,
            kernel_size,
            strides=(1, 1),
            padding='valid',
            data_format='channels_last',
            activation=None,
            use_bias=True,
            kernel_initializer=None,
            bias_initializer=tf.zeros_initializer(),
            kernel_regularizer=None,
            bias_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            bias_constraint=None,
            trainable=True,
            name=None,
            reuse=None
        )
        """
        if upsampling == 'DeConv':
            return tf.layers.conv2d_transpose(inputs=inputs,
                                              filters=kwargs['output_channel'],
                                              kernel_size=kwargs['kernel_size'],
                                              strides=kwargs['strides'],
                                              padding=kwargs['padding'],
                                              activation=kwargs['activation'],
                                              name=kwargs['name'])  # '3 3 2 2 same ReLU'
        if upsampling == 'BiLi':
            return tf.image.resize_images(inputs, size=(kwargs['size'], kwargs['size']))  # '3'

        raise ValueError('No such UpSampling methods.')


class VGG16(BaseNet):
    def net_loading(self, **kwargs):
        """
        :param input_image: image to be fed forward
                            tensor of size  batch_size * height * width * channels
        :param pooling:
        :return: a dict containing activations of all layers
        """
        with tf.variable_scope(tf.get_variable_scope(), custom_getter=cpu_variable_getter):

            input_image = kwargs['input_image']

            self.layer = [
                'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',  # part1
                'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',  # part2
                'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
                'relu3_3', 'pool3',                                   # part3
                'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
                'relu4_3', 'pool4',                                   # part4
                'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
                'relu5_3', 'pool5'                                    # part5
            ]

            filter_shapes = {
                'conv1_1': (3, 3, 3, 64), 'conv1_2': (3, 3, 64, 64),                      # part1
                'conv2_1': (3, 3, 64, 128), 'conv2_2': (3, 3, 128, 128),                      # part2
                'conv3_1': (3, 3, 128, 256), 'conv3_2': (3, 3, 256, 256), 'conv3_3': (3, 3, 256, 256),           # part3
                'conv4_1': (3, 3, 256, 512), 'conv4_2': (3, 3, 512, 512), 'conv4_3': (3, 3, 512, 512),           # part4
                'conv5_1': (3, 3, 512, 512), 'conv5_2': (3, 3, 512, 512), 'conv5_3': (3, 3, 512, 512)            # part4
            }

            net = {}

            activations = self.pre_processing(input_image)
            for i, name in enumerate(self.layer):
                Layer_type = name[:4]
                if Layer_type == 'conv':
                    activations = BaseNet.ConvLayer(activations, shape=filter_shapes[name], padding=self.padding, name=name)
                elif Layer_type == 'relu':
                    activations = tf.nn.relu(activations)
                elif Layer_type == 'pool':
                    activations = BaseNet.Pooling(activations, self.pooling, name=name)
                    net['stage' + name[-1]] = activations = tf.nn.l2_normalize(activations, dim=-1, name='norm') * 20  # optional, not in the original VGG network
                net[name] = activations

            self.log['info']("BackBone network built.")
            self.pipe = {"input_pipe": input_image,
                         "output_pipe": net}


class VGG19(BaseNet):
    def net_loading(self, **kwargs):
        """
        :param weights: weights for the ConvNet
        :param input_image: image to be fed forward#tensor of size  batch_size * height * width * channels
        :param pooling:
        :return: a dict containing activations of all layers
        """
        with tf.variable_scope(tf.get_variable_scope(), custom_getter=cpu_variable_getter):
            input_image = kwargs['input_image']
            self.layer = [
                'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',  # part1
                'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',  # part2
                'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
                'relu3_3', 'conv3_4', 'relu3_4', 'pool3',             # part3
                'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
                'relu4_3', 'conv4_4', 'relu4_4', 'pool4',             # part4
                'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
                'relu5_3', 'conv5_4', 'relu5_4', 'pool5'              # part5
            ]

            filter_shapes = {
                'conv1_1': (3, 3, 3, 64), 'conv1_2': (3, 3, 64, 64),        # part1
                'conv2_1': (3, 3, 64, 128), 'conv2_2': (3, 3, 128, 128),    # part2
                'conv3_1': (3, 3, 128, 256), 'conv3_2': (3, 3, 256, 256), 'conv3_3': (3, 3, 256, 256), 'conv3_4': (3, 3, 256, 256),  # part3
                'conv4_1': (3, 3, 256, 512), 'conv4_2': (3, 3, 512, 512), 'conv4_3': (3, 3, 512, 512), 'conv4_4': (3, 3, 512, 512),  # part4
                'conv5_1': (3, 3, 512, 512), 'conv5_2': (3, 3, 512, 512), 'conv5_3': (3, 3, 512, 512), 'conv5_4': (3, 3, 512, 512)   # part4
            }

            net = {}

            activations = self.pre_processing(input_image)
            for i, name in enumerate(self.layer):
                Layer_type = name[:4]
                if Layer_type == 'conv':
                    activations = BaseNet.ConvLayer(activations, shape=filter_shapes[name], padding=self.padding, name=name)
                elif Layer_type == 'relu':
                    activations = tf.nn.relu(activations)
                elif Layer_type == 'pool':
                    activations = BaseNet.Pooling(activations, self.pooling, name=name)
                    net['stage' + name[-1]] = activations = tf.nn.l2_normalize(activations, dim=-1,
                                                                               name='norm') * 20  # optional, not in the original VGG network
                net[name] = activations

            self.log['info']("BackBone network built.")
            self.pipe = {"input_pipe": input_image,
                         "output_pipe": net}


class ResNet(BaseNet):
    def net_loading(self, **kwargs):
        """
        :param input_image:
        :param layer: int, 18, 34, 50, 101, 152
        :return:
        """
        with tf.variable_scope(tf.get_variable_scope(), custom_getter=cpu_variable_getter):
            input_image = kwargs['input_image']
            layer = int(kwargs['layer'])
            if not layer in ResNetSchedule.layouts:
                raise ValueError('no such resnet layout')
            layout = ResNetSchedule.layouts[layer]

            activations = self.pre_processing(input_image)
            net = {}

            # for stage 1

            activations = BaseNet.ConvLayer(activations, shape=(7, 7, 3, 64), padding=self.padding, strides=(1, 2, 2, 1), name='stage1')

            net['stage1'] = activations

            for i, stage in enumerate(['stage2', 'stage3', 'stage4', 'stage5']):
                structure = layout[i]
                for block in range(structure[1]):
                    if block == 0 and stage == 'stage2':
                        block_acti = BaseNet.Pooling(activations, 'max', name='Stage2Pooling')
                    else:
                        block_acti = activations

                    for step, filter_shape in enumerate(structure[0]):  # filter_shape (k1,k2,output_channel)
                        if stage != 'stage2' and block == 0 and step == 0:
                            strides = (1, 2, 2, 1)
                        else:
                            strides = (1, 1, 1, 1)
                        shape = block_acti.shape.as_list()
                        block_acti = BaseNet.ConvLayer(block_acti, shape=(
                            filter_shape[0], filter_shape[1], shape[-1], filter_shape[2]),
                            padding=self.padding, strides=strides,
                            name=stage + '_' + 'block_' + str(block + 2) + 'step_' + str(step))
                        if step == len(structure[0]) - 1:
                            break
                        block_acti = tf.nn.relu(block_acti)
                    shape_acti = activations.shape.as_list()
                    shape_block = block_acti.shape.as_list()

                    if block == 0:
                        pooled = BaseNet.Pooling(activations,
                                                 pooling='avg',
                                                 name=stage + '_block_' + str(block) + 'Pooling')
                        activations = block_acti + tf.concat([pooled] + [pooled * 0] * (shape_block[-1] // shape_acti[-1] - 1),
                                                             axis=-1)
                    else:
                        activations = block_acti + activations
                    activations = tf.nn.relu(activations)

                net[stage] = activations

            self.log['info']("BackBone network built.")
            self.pipe = {"input_pipe": input_image,
                         "output_pipe": net}


#tf.get_variable(name, initializer=filters, dtype=tf.float32, trainable=trainable)
