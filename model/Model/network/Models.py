from . import Basenet
import tensorflow as tf
import os
import time
from .yellowfin import *


def average_gradients(grads):
    average_grads = []
    for grad_and_vars in zip(*grads):
        grads = []
        for g, _ in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)

        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)

        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)

    return average_grads


def flatten(output):
    """
    :param output: Dict(layer=tensor[batch,h,w,2]) for score map only
    :return: tensor[batch,C]
    """
    shape = output.shape.as_list()
    if shape[-1] == 2:
        return tf.reshape(output, shape=(-1, 2))
    else:
        return tf.reshape(output, shape=(-1, 1))


def smooth_l1_loss(pred, target):
    diff = pred - target
    abs_diff = tf.abs(diff)
    abs_diff_lt_1 = tf.less(abs_diff, 1)
    loss = tf.where(abs_diff_lt_1, 0.5 * tf.square(abs_diff), abs_diff - 0.5)
    return tf.reduce_sum(loss)


class network_multi_gpu(object):
    def __init__(self, configs, logs):
        self.configs = configs
        self.logs = logs
        self._build_network()
        self.saver = tf.train.Saver(var_list=tf.trainable_variables(),
                                    max_to_keep=5)

    # this function is to build the network stem from BaseNet
    def _build_network(self):
        with tf.device('/device:cpu:0'):
            self.input_image = tf.placeholder(tf.float32,
                                              shape=(None,
                                                     None, None, 3),
                                              name="InputImage")
            self.Labels = tf.placeholder(tf.float32,
                                         shape=(None,
                                                None, None, 5),
                                         name="Labels")
            self.LossWeights = tf.placeholder(tf.float32,
                                              shape=(None,
                                                     None, None, 1),
                                              name="LossWeights")

            input_split = tf.split(self.input_image, len(self.configs.gpu_list), axis=0)
            Labels_split = tf.split(self.Labels, len(self.configs.gpu_list), axis=0)
            self.global_step = tf.get_variable('global_step', initializer=tf.constant(0, dtype=tf.int32), trainable=False)
            grads_per_gpu = []
            reuse_variables = None
            opt = self._train()
            self.prediction = []
            for i, gpu_id in enumerate(self.configs.gpu_list):
                self.logs['info']('building network for gpu:%s' % gpu_id)
                with tf.device('/device:GPU:%s' % i):
                    with tf.name_scope('model_%s' % gpu_id) as scope:
                        # if True:
                        input_per_gpu = input_split[i]
                        labels_per_gpu = Labels_split[i]
                        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
                            pipe = self._build_back_bone(input_per_gpu)
                            prediction = self._add_prediction_block(pipe)
                            self.prediction.append(prediction)
                            self.total_loss, score_loss, geo_loss, geo_attr, TR_score_loss = self._build_loss(labels_per_gpu, prediction)
                            grads = opt.compute_gradients(self.total_loss)
                            self.logs['info']('prediction layer / loss computation finished')
                            grads_per_gpu.append(grads)
                        if reuse_variables is None:
                            TCL_Loss = tf.summary.scalar("Text_center_line_map_Loss", score_loss)
                            TR_Loss = tf.summary.scalar("Text_Region_map_Loss", TR_score_loss)
                            Loss = tf.summary.scalar("total_loss_Loss", self.total_loss)
                            tf.summary.histogram('cos_his', self.prediction[0][0:1, :, :, 3:4])
                            tf.summary.histogram('cos_his_GT', self.Labels[0:1, :, :, 2:3])
                            tf.summary.histogram('sin_his', self.prediction[0][0:1, :, :, 4:5])
                            tf.summary.histogram('sin_his_GT', self.Labels[0:1, :, :, 3:4])

                            GEO_LOSS = []

                            for i in range(len(geo_attr)):
                                GEO_LOSS.append(tf.summary.scalar('%s_loss' % geo_attr[i], geo_loss[i]))

                            self.summary_op = tf.summary.merge(
                                [TCL_Loss, TR_Loss, Loss] + GEO_LOSS)

                            tf.summary.scalar("Train_Loss", self.total_loss)

                            # TR/TCL map
                            self.TR_TCL_GT_map = tf.concat([self.Labels[:, :, :, 4:5],  # TR
                                                            self.Labels[:, :, :, 0:1],  # TCL
                                                            self.Labels[:, :, :, 4:5] * 0], axis=-1)  # TODO: add new predicting layers: start&end point of text lines
                            L_sum = tf.summary.image('TR_TCL_GT_map', self.TR_TCL_GT_map[0:1, :, :, :], 1)
                            # TR/TCL predicted map
                            self.TR_TCL_Predictied_map = tf.concat([tf.nn.softmax(logits=self.prediction[0][:, :, :, 5:7], dim=-1)[:, :, :, 1:2],
                                                                    tf.cast(self.prediction[0][:, :, :, 6:7] >
                                                                            self.prediction[0][:, :, :, 5:6], tf.float32) *
                                                                    (tf.nn.softmax(logits=self.prediction[0][:, :, :, 0:2], dim=-1)[:, :, :, 1:2]),
                                                                    self.prediction[0][:, :, :, 5:6] * 0],
                                                                   axis=-1)
                            P_sum = tf.summary.image('TR_TCL_Predictied_map', self.TR_TCL_Predictied_map[0:1, :, :, :], 1)
                            # R map
                            max_light = tf.reduce_max(self.Labels[:, :, :, 1:2])
                            self.L_R_map = tf.concat([self.Labels[:, :, :, 1:2],
                                                      self.Labels[:, :, :, 1:2],
                                                      self.Labels[:, :, :, 1:2]], axis=-1) / max_light * 200 / 255
                            self.P_R_map = tf.concat([self.prediction[0][:, :, :, 2:3],
                                                      self.prediction[0][:, :, :, 2:3],
                                                      self.prediction[0][:, :, :, 2:3]], axis=-1) / max_light * 200 / 255

                            L_R_sum = tf.summary.image('Radius_L_map', self.L_R_map[0:1, :, :, :], 1)
                            P_R_sum = tf.summary.image('Radius_P_map', self.P_R_map[0:1, :, :, :], 1)

                            # sin map
                            self.L_sin = tf.concat([(self.Labels[:, :, :, 3:4] + 1) / 2,
                                                    (self.Labels[:, :, :, 3:4] + 1) / 2,
                                                    (self.Labels[:, :, :, 3:4] + 1) / 2], axis=-1)
                            self.P_sin = tf.concat([(self.prediction[0][:, :, :, 4:5] + 1) / 2,
                                                    (self.prediction[0][:, :, :, 4:5] + 1) / 2,
                                                    (self.prediction[0][:, :, :, 4:5] + 1) / 2], axis=-1)
                            L_sin_sum = tf.summary.image('sin_L_map', self.L_sin, 1)
                            P_sin_sum = tf.summary.image('sin_P_map', self.P_sin, 1)

                            # cos map
                            self.L_cos = tf.concat([self.Labels[:, :, :, 2:3],
                                                    self.Labels[:, :, :, 2:3],
                                                    self.Labels[:, :, :, 2:3]], axis=-1)
                            self.P_cos = tf.concat([self.prediction[0][:, :, :, 3:4],
                                                    self.prediction[0][:, :, :, 3:4],
                                                    self.prediction[0][:, :, :, 3:4]], axis=-1)
                            L_cos_sum = tf.summary.image('cos_L_map', self.L_cos[0:1, :, :, :], 1)
                            P_cos_sum = tf.summary.image('cos_P_map', self.P_cos[0:1, :, :, :], 1)

                            im_sum = tf.summary.image('x_original_image', self.input_image[0:1, :, :, ::-1],
                                                      1)  # printing only the first pic

                            self.image_summary_op = tf.summary.merge(
                                [L_sum, P_sum, im_sum])
                            self.image_attributes_summary_op = tf.summary.merge(
                                [L_R_sum, P_R_sum, L_sin_sum, P_sin_sum, L_cos_sum, P_cos_sum])
                        reuse_variables = True

            grads = average_gradients(grads_per_gpu)
            if self.configs.moving_average_decay < 1:
                variable_averages = tf.train.ExponentialMovingAverage(
                    self.configs.moving_average_decay,
                    self.global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)
                with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
                    self.train_step = tf.no_op(name='train_op')
            else:
                self.train_step = opt.apply_gradients(grads, global_step=self.global_step)
            self.prediction = tf.concat(self.prediction, axis=0)

    def _build_back_bone(self, input_image):
        basenets = {'vgg16': Basenet.VGG16, 'vgg19': Basenet.VGG16, 'resnet': Basenet.ResNet}  # for resnet :  'resnet-layer_number'
        basenet = self.basenet = basenets[self.configs.basenet[:self.configs.basenet.find('-')]](self.configs, self.logs) \
            if self.configs.basenet.startswith('resnet') \
            else basenets[self.configs.basenet](self.configs, self.logs)
        basenet.net_loading(input_image=input_image, layer=self.configs.basenet[self.configs.basenet.find('-') + 1:])
        return basenet.pipe

    # this function is to build blocks for predictions of pixels
    def _add_prediction_block(self, pipe):
        # prediciton stage
        params = self.configs.US_Params.split(' ')
        activation_functions = {'None': None, 'ReLU': tf.nn.relu, 'cReLU': tf.nn.crelu, 'softmax': tf.nn.softmax}
        if self.configs.upsampling == 'DeConv':
            kwargs = {'output_channel': None,
                      'kernel_size': (int(params[0]), int(params[1])),
                      'strides': (int(params[2]), int(params[3])),
                      'padding': params[4],
                      'activation': activation_functions[params[5]],
                      'name': None}
        else:
            kwargs = {
                'size': int(params)
            }
        pipe['prediction'] = {}
        # stage 1
        pipe['prediction']['stage1'] = h = pipe['output_pipe']['stage5']

        # stage 2-4
        for stage in range(2, self.configs.Predict_stage + 1):  # [2,5)
            kwargs['output_channel'] = h.shape.as_list()[-1] // 2
            kwargs['name'] = 'stage%d_US' % stage
            h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
            h = tf.concat([h, pipe['output_pipe']['stage' + str(6 - stage)]], axis=-1)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              1, 1, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_1')
            h = tf.nn.relu(h)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              3, 3, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_2')

            pipe['prediction']['stage' + str(stage)] = h = tf.nn.relu(h)

        # stage 5-6
        kwargs['output_channel'] = self.configs.predict_channels[-1]
        kwargs['name'] = 'stage%d_US' % (-2)
        h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
        kwargs['name'] = 'stage%d_US' % (-1)
        h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
        h = Basenet.BaseNet.ConvLayer(h,
                                      shape=(
                                          3, 3, h.shape.as_list()[-1],
                                          self.configs.predict_channels[self.configs.Predict_stage - 1]),
                                      padding=self.configs.padding, strides=(1, 1, 1, 1),
                                      name='Predict_stage_penul')
        h = tf.nn.relu(h)

        # final stage to make prediction
        prediction = h = Basenet.BaseNet.ConvLayer(h,
                                                   shape=(
                                                       3, 3, h.shape.as_list()[-1],  # TODO kernel size should be (1,1) here(?)
                                                       self.configs.Label_size[2] + 1 + 1),
                                                   # plus one for text region map
                                                   padding=self.configs.padding, strides=(1, 1, 1, 1),
                                                   name='Predict_stage_final')  # ,

        # regularize cos and sin to a squared sum of 1
        cos_sin_sum = tf.sqrt(prediction[:, :, :, 3] * prediction[:, :, :, 3] + prediction[:, :, :, 4] * prediction[:, :, :, 4])
        cos = prediction[:, :, :, 3] / cos_sin_sum
        sin = prediction[:, :, :, 4] / cos_sin_sum
        return tf.stack([prediction[:, :, :, 0], prediction[:, :, :, 1], prediction[:, :, :, 2], cos, sin, prediction[:, :, :, 5], prediction[:, :, :, 6]], axis=-1)
        # TODO at present, the score map is predicted in two channels (1 for pos, 1 for neg), we can also consider sigmoid loss for one channel. but this might be trivial

    # this function is to build loss function
    def _build_loss(self, Labels, prediction):
        with tf.name_scope('TR_loss'):
            def pos_mask_TR():
                # from self.Labels[:,:,:,0:1]
                return Labels[:, :, :, 4:5] > 0
            pos_TR = tf.cast(pos_mask_TR(), tf.float32)
            pos_num_TR = tf.reduce_sum(pos_TR) + 1e-3
            neg_num_TR = tf.cast(3 * pos_num_TR + 1, tf.int32)  # in case, OHNM is used
            # TR score map loss
            singel_labels_TR = flatten(Labels[:, :, :, 4:5])
            one_hot_labels_TR = tf.concat([1 - singel_labels_TR, singel_labels_TR], axis=-1)
            loss_TR = tf.losses.softmax_cross_entropy(one_hot_labels_TR,
                                                      flatten(prediction[:, :, :, 5:7]),
                                                      reduction=tf.losses.Reduction.NONE)
            pos_flatten_TR = tf.reshape(flatten(pos_TR), shape=(-1,))

            pos_loss_TR = loss_TR * pos_flatten_TR
            neg_losses_TR = loss_TR * (1 - pos_flatten_TR)
            neg_loss_TR = tf.nn.top_k(neg_losses_TR, k=tf.reduce_min((neg_num_TR, tf.size(neg_losses_TR)))).values
            TR_score_loss = (tf.reduce_sum(pos_loss_TR) / pos_num_TR +
                             tf.reduce_sum(neg_loss_TR) / pos_num_TR)  # top_k in use

        with tf.name_scope('TCL_loss'):
            def pos_mask():
                # from self.Labels[:,:,:,0:1]
                return Labels[:, :, :, 0:1] > 0
            pos = tf.cast(pos_mask(), tf.float32)
            pos_num = tf.reduce_sum(pos) + 1e-3
            neg_num = tf.cast(3 * pos_num + 1, tf.int32)  # in case, OHNM is used
            # score map loss
            singel_labels = flatten(Labels[:, :, :, 0:1])
            one_hot_labels = tf.concat([1 - singel_labels, singel_labels], axis=-1)
            loss = tf.losses.softmax_cross_entropy(one_hot_labels,
                                                   flatten(prediction[:, :, :, 0:2]),
                                                   reduction=tf.losses.Reduction.NONE)

            pos_flatten = tf.reshape(flatten(pos), shape=(-1,))
            pos_loss = loss * pos_flatten * pos_flatten_TR
            neg_loss = loss * (1 - pos_flatten) * pos_flatten_TR
            # neg_loss=tf.nn.top_k(neg_losses,k=neg_num).values
            score_loss = (tf.reduce_sum(pos_loss) / pos_num +
                          tf.reduce_sum(neg_loss) / pos_num)  # top_k not in use TODO: rebalance the pos/neg number !

        with tf.name_scope('Geo_loss'):
            geo_attr = ['radius', 'cos', 'sin']
            geo_loss = []
            total_loss = score_loss + TR_score_loss  # for training
            for i in range(4 - 1):  # self.configs.Label_size[2]-1):
                geo_loss.append(smooth_l1_loss(flatten(Labels[:, :, :, i + 1:i + 2] * pos),
                                               flatten(prediction[:, :, :, i + 2:i + 3] * pos)
                                               ) / pos_num)
                total_loss += geo_loss[-1]

        return total_loss, score_loss, geo_loss, geo_attr, TR_score_loss

    def _train(self):
        optimizer = {'Adam': tf.train.AdamOptimizer, 'YF': YFOptimizer, 'SGD': tf.train.GradientDescentOptimizer}
        return optimizer[self.configs.optimizer](learning_rate=self._learning_rate())

    def save(self, sess, path=None):
        if path:
            self.saver.save(sess, path)
        else:
            self.saver.save(sess, os.path.join(path, str(int(time.time())) + '.cntk'))

    def _learning_rate(self):
        decay_type = {'expo': tf.train.exponential_decay,
                      'inverse': tf.train.inverse_time_decay,
                      'nat': tf.train.natural_exp_decay,
                      'pln': tf.train.polynomial_decay,
                      'None': lambda *args: self.configs.learning_rate}
        return tf.minimum(decay_type[self.configs.decay_type](self.configs.learning_rate, self.global_step, *self.configs.decay_params),
                          self.configs.learning_rate / 100)

    def load(self, sess, path):
        try:
            self.saver.restore(sess, path)
        except:
            self.logs['debug']('failed to restore.')
            raise ValueError('failed to restore.')


class network_output_128(network_multi_gpu):
    def _build_network(self):
        with tf.device('/device:cpu:0'):
            self.input_image = tf.placeholder(tf.float32,
                                              shape=(None,
                                                     None, None, 3),
                                              name="InputImage")
            self.Labels = tf.placeholder(tf.float32,
                                         shape=(None,
                                                None, None, 5),
                                         name="Labels")
            self.resized_Labels = tf.concat([Basenet.BaseNet.Pooling(self.Labels[:, :, :, 0:1], 'max', name='input_resize_1', strides=(1, 4, 4, 1), padding='VALID', ksize=(1, 4, 4, 1)),
                                             Basenet.BaseNet.Pooling(self.Labels[:, :, :, 1:4], 'avg', name='input_resize_2', strides=(1, 4, 4, 1), padding='VALID', ksize=(1, 4, 4, 1)),
                                             Basenet.BaseNet.Pooling(self.Labels[:, :, :, 4:5], 'max', name='input_resize_3', strides=(1, 4, 4, 1), padding='VALID', ksize=(1, 4, 4, 1))],
                                            axis=-1)
            self.LossWeights = tf.placeholder(tf.float32,
                                              shape=(None,
                                                     None, None, 1),
                                              name="LossWeights")

            input_split = tf.split(self.input_image, len(self.configs.gpu_list), axis=0)
            Labels_split = tf.split(self.resized_Labels, len(self.configs.gpu_list), axis=0)
            self.global_step = tf.get_variable('global_step', initializer=tf.constant(0, dtype=tf.int32), trainable=False)
            grads_per_gpu = []
            reuse_variables = None
            opt = self._train()
            self.prediction = []
            for i, gpu_id in enumerate(self.configs.gpu_list):
                self.logs['info']('building network for gpu:%s' % gpu_id)
                with tf.device('/device:GPU:%s' % i):
                    with tf.name_scope('model_%s' % gpu_id) as scope:
                        # if True:
                        input_per_gpu = input_split[i]
                        labels_per_gpu = Labels_split[i]
                        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
                            pipe = self._build_back_bone(input_per_gpu)
                            prediction = self._add_prediction_block(pipe)
                            self.prediction.append(prediction)
                            self.total_loss, score_loss, geo_loss, geo_attr, TR_score_loss = self._build_loss(labels_per_gpu, prediction)
                            grads = opt.compute_gradients(self.total_loss)
                            self.logs['info']('prediction layer / loss computation finished')
                            grads_per_gpu.append(grads)
                        if reuse_variables is None:
                            TCL_Loss = tf.summary.scalar("Text_center_line_map_Loss", score_loss)
                            TR_Loss = tf.summary.scalar("Text_Region_map_Loss", TR_score_loss)
                            Loss = tf.summary.scalar("total_loss_Loss", self.total_loss)
                            tf.summary.histogram('cos_his', self.prediction[0][0:1, :, :, 3:4])
                            tf.summary.histogram('cos_his_GT', self.resized_Labels[0:1, :, :, 2:3])
                            tf.summary.histogram('sin_his', self.prediction[0][0:1, :, :, 4:5])
                            tf.summary.histogram('sin_his_GT', self.resized_Labels[0:1, :, :, 3:4])

                            GEO_LOSS = []

                            for i in range(len(geo_attr)):
                                GEO_LOSS.append(tf.summary.scalar('%s_loss' % geo_attr[i], geo_loss[i]))

                            self.summary_op = tf.summary.merge(
                                [TCL_Loss, TR_Loss, Loss] + GEO_LOSS)

                            tf.summary.scalar("Train_Loss", self.total_loss)

                            # TR/TCL map
                            self.TR_TCL_GT_map = tf.concat([self.resized_Labels[:, :, :, 4:5],  # TR
                                                            self.resized_Labels[:, :, :, 0:1],  # TCL
                                                            self.resized_Labels[:, :, :, 4:5] * 0], axis=-1)  # TODO: add new predicting layers: start&end point of text lines
                            L_sum = tf.summary.image('TR_TCL_GT_map', self.TR_TCL_GT_map[0:1, :, :, :], 1)
                            # TR/TCL predicted map
                            self.TR_TCL_Predictied_map = tf.concat([tf.nn.softmax(logits=self.prediction[0][:, :, :, 5:7], dim=-1)[:, :, :, 1:2],
                                                                    tf.cast(self.prediction[0][:, :, :, 6:7] >
                                                                            self.prediction[0][:, :, :, 5:6], tf.float32) *
                                                                    (tf.nn.softmax(logits=self.prediction[0][:, :, :, 0:2], dim=-1)[:, :, :, 1:2]),
                                                                    self.prediction[0][:, :, :, 5:6] * 0],
                                                                   axis=-1)
                            P_sum = tf.summary.image('TR_TCL_Predictied_map', self.TR_TCL_Predictied_map[0:1, :, :, :], 1)
                            # R map
                            max_light = tf.reduce_max(self.resized_Labels[:, :, :, 1:2])
                            self.L_R_map = tf.concat([self.resized_Labels[:, :, :, 1:2],
                                                      self.resized_Labels[:, :, :, 1:2],
                                                      self.resized_Labels[:, :, :, 1:2]], axis=-1) / max_light * 200 / 255
                            self.P_R_map = tf.concat([self.prediction[0][:, :, :, 2:3],
                                                      self.prediction[0][:, :, :, 2:3],
                                                      self.prediction[0][:, :, :, 2:3]], axis=-1) / max_light * 200 / 255

                            L_R_sum = tf.summary.image('Radius_L_map', self.L_R_map[0:1, :, :, :], 1)
                            P_R_sum = tf.summary.image('Radius_P_map', self.P_R_map[0:1, :, :, :], 1)

                            # sin map
                            self.L_sin = tf.concat([(self.resized_Labels[:, :, :, 3:4] + 1) / 2,
                                                    (self.resized_Labels[:, :, :, 3:4] + 1) / 2,
                                                    (self.resized_Labels[:, :, :, 3:4] + 1) / 2], axis=-1)
                            self.P_sin = tf.concat([(self.prediction[0][:, :, :, 4:5] + 1) / 2,
                                                    (self.prediction[0][:, :, :, 4:5] + 1) / 2,
                                                    (self.prediction[0][:, :, :, 4:5] + 1) / 2], axis=-1)
                            L_sin_sum = tf.summary.image('sin_L_map', self.L_sin, 1)
                            P_sin_sum = tf.summary.image('sin_P_map', self.P_sin, 1)

                            # cos map
                            self.L_cos = tf.concat([self.resized_Labels[:, :, :, 2:3],
                                                    self.resized_Labels[:, :, :, 2:3],
                                                    self.resized_Labels[:, :, :, 2:3]], axis=-1)
                            self.P_cos = tf.concat([self.prediction[0][:, :, :, 3:4],
                                                    self.prediction[0][:, :, :, 3:4],
                                                    self.prediction[0][:, :, :, 3:4]], axis=-1)
                            L_cos_sum = tf.summary.image('cos_L_map', self.L_cos[0:1, :, :, :], 1)
                            P_cos_sum = tf.summary.image('cos_P_map', self.P_cos[0:1, :, :, :], 1)

                            im_sum = tf.summary.image('x_original_image', self.input_image[0:1, :, :, ::-1],
                                                      1)  # printing only the first pic

                            self.image_summary_op = tf.summary.merge(
                                [L_sum, P_sum, im_sum])
                            self.image_attributes_summary_op = tf.summary.merge(
                                [L_R_sum, P_R_sum, L_sin_sum, P_sin_sum, L_cos_sum, P_cos_sum])
                        reuse_variables = True

            grads = average_gradients(grads_per_gpu)
            if self.configs.moving_average_decay < 1:
                variable_averages = tf.train.ExponentialMovingAverage(
                    self.configs.moving_average_decay,
                    self.global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)
                with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
                    self.train_step = tf.no_op(name='train_op')
            else:
                self.train_step = opt.apply_gradients(grads, global_step=self.global_step)
            self.prediction = tf.concat(self.prediction, axis=0)

    def _add_prediction_block(self, pipe):
        # prediciton stage
        params = self.configs.US_Params.split(' ')
        activation_functions = {'None': None, 'ReLU': tf.nn.relu, 'cReLU': tf.nn.crelu, 'softmax': tf.nn.softmax}
        if self.configs.upsampling == 'DeConv':
            kwargs = {'output_channel': None,
                      'kernel_size': (int(params[0]), int(params[1])),
                      'strides': (int(params[2]), int(params[3])),
                      'padding': params[4],
                      'activation': activation_functions[params[5]],
                      'name': None}
        else:
            kwargs = {
                'size': int(params)
            }
        pipe['prediction'] = {}
        # stage 1
        pipe['prediction']['stage1'] = h = pipe['output_pipe']['stage5']

        # stage 2-4
        for stage in range(2, self.configs.Predict_stage + 1):  # [2,5)
            kwargs['output_channel'] = h.shape.as_list()[-1] // 2
            kwargs['name'] = 'stage%d_US' % stage
            h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
            h = tf.concat([h, pipe['output_pipe']['stage' + str(6 - stage)]], axis=-1)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              1, 1, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_1')
            h = tf.nn.relu(h)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              3, 3, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_2')

            pipe['prediction']['stage' + str(stage)] = h = tf.nn.relu(h)

        # penultimate stage
        h = Basenet.BaseNet.ConvLayer(h,
                                      shape=(
                                          3, 3, h.shape.as_list()[-1],
                                          self.configs.predict_channels[self.configs.Predict_stage - 1]),
                                      padding=self.configs.padding, strides=(1, 1, 1, 1),
                                      name='Predict_stage_penul')
        h = tf.nn.relu(h)

        # final stage to make prediction
        prediction = h = Basenet.BaseNet.ConvLayer(h,
                                                   shape=(
                                                       3, 3, h.shape.as_list()[-1],  # TODO kernel size should be (1,1) here(?)
                                                       self.configs.Label_size[2] + 1 + 1),
                                                   # plus one for text region map
                                                   padding=self.configs.padding, strides=(1, 1, 1, 1),
                                                   name='Predict_stage_final')  # ,

        # regularize cos and sin to a squared sum of 1
        cos_sin_sum = tf.sqrt(prediction[:, :, :, 3] * prediction[:, :, :, 3] + prediction[:, :, :, 4] * prediction[:, :, :, 4])
        cos = prediction[:, :, :, 3] / cos_sin_sum
        sin = prediction[:, :, :, 4] / cos_sin_sum
        return tf.stack([prediction[:, :, :, 0], prediction[:, :, :, 1], prediction[:, :, :, 2], cos, sin, prediction[:, :, :, 5], prediction[:, :, :, 6]], axis=-1)
        # TODO at present, the score map is predicted in two channels (1 for pos, 1 for neg), we can also consider sigmoid loss for one channel. but this might be trivial


class network_output_kennel_size_1(network_multi_gpu):
    # this function is to build blocks for predictions of pixels
    def _add_prediction_block(self, pipe):
        # prediciton stage
        params = self.configs.US_Params.split(' ')
        activation_functions = {'None': None, 'ReLU': tf.nn.relu, 'cReLU': tf.nn.crelu, 'softmax': tf.nn.softmax}
        if self.configs.upsampling == 'DeConv':
            kwargs = {'output_channel': None,
                      'kernel_size': (int(params[0]), int(params[1])),
                      'strides': (int(params[2]), int(params[3])),
                      'padding': params[4],
                      'activation': activation_functions[params[5]],
                      'name': None}
        else:
            kwargs = {
                'size': int(params)
            }
        pipe['prediction'] = {}
        # stage 1
        pipe['prediction']['stage1'] = h = pipe['output_pipe']['stage5']

        # stage 2-4
        for stage in range(2, self.configs.Predict_stage + 1):  # [2,5)
            kwargs['output_channel'] = h.shape.as_list()[-1] // 2
            kwargs['name'] = 'stage%d_US' % stage
            h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
            h = tf.concat([h, pipe['output_pipe']['stage' + str(6 - stage)]], axis=-1)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              1, 1, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_1')
            h = tf.nn.relu(h)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              3, 3, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_2')

            pipe['prediction']['stage' + str(stage)] = h = tf.nn.relu(h)

        # stage 5-6
        kwargs['output_channel'] = self.configs.predict_channels[-1]
        kwargs['name'] = 'stage%d_US' % (-2)
        h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
        kwargs['name'] = 'stage%d_US' % (-1)
        h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
        h = Basenet.BaseNet.ConvLayer(h,
                                      shape=(
                                          3, 3, h.shape.as_list()[-1],
                                          self.configs.predict_channels[self.configs.Predict_stage - 1]),
                                      padding=self.configs.padding, strides=(1, 1, 1, 1),
                                      name='Predict_stage_penul')
        h = tf.nn.relu(h)

        # final stage to make prediction
        prediction = h = Basenet.BaseNet.ConvLayer(h,
                                                   shape=(
                                                       1, 1, h.shape.as_list()[-1],
                                                       self.configs.Label_size[2] + 1 + 1),
                                                   # plus one for text region map
                                                   padding=self.configs.padding, strides=(1, 1, 1, 1),
                                                   name='Predict_stage_final')  # ,

        # regularize cos and sin to a squared sum of 1
        cos_sin_sum = tf.sqrt(prediction[:, :, :, 3] * prediction[:, :, :, 3] + prediction[:, :, :, 4] * prediction[:, :, :, 4])
        cos = prediction[:, :, :, 3] / cos_sin_sum
        sin = prediction[:, :, :, 4] / cos_sin_sum
        return tf.stack([prediction[:, :, :, 0], prediction[:, :, :, 1], prediction[:, :, :, 2], cos, sin, prediction[:, :, :, 5], prediction[:, :, :, 6]], axis=-1)
        # TODO at present, the score map is predicted in two channels (1 for pos, 1 for neg), we can also consider sigmoid loss for one channel. but this might be trivial


class network_output_down_to_stage_1_deconv_1(network_multi_gpu):
    # this function is to build blocks for predictions of pixels
    def _add_prediction_block(self, pipe):
        # prediciton stage
        params = self.configs.US_Params.split(' ')
        activation_functions = {'None': None, 'ReLU': tf.nn.relu, 'cReLU': tf.nn.crelu, 'softmax': tf.nn.softmax}
        if self.configs.upsampling == 'DeConv':
            kwargs = {'output_channel': None,
                      'kernel_size': (int(params[0]), int(params[1])),
                      'strides': (int(params[2]), int(params[3])),
                      'padding': params[4],
                      'activation': activation_functions[params[5]],
                      'name': None}
        else:
            kwargs = {
                'size': int(params)
            }
        pipe['prediction'] = {}
        # stage 1
        pipe['prediction']['stage1'] = h = pipe['output_pipe']['stage5']

        # stage 2-4
        for stage in range(2, self.configs.Predict_stage + 1 + 1):  # [2,5)
            kwargs['output_channel'] = h.shape.as_list()[-1] // 2
            kwargs['name'] = 'stage%d_US' % stage
            h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
            h = tf.concat([h, pipe['output_pipe']['stage' + str(6 - stage)]], axis=-1)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              1, 1, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_1')
            h = tf.nn.relu(h)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              3, 3, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_2')

            pipe['prediction']['stage' + str(stage)] = h = tf.nn.relu(h)

        # stage 5-6
        kwargs['output_channel'] = self.configs.predict_channels[-1]
        kwargs['name'] = 'stage%d_US' % (-2)
        h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
        h = Basenet.BaseNet.ConvLayer(h,
                                      shape=(
                                          3, 3, h.shape.as_list()[-1],
                                          self.configs.predict_channels[self.configs.Predict_stage - 1]),
                                      padding=self.configs.padding, strides=(1, 1, 1, 1),
                                      name='Predict_stage_penul')
        h = tf.nn.relu(h)

        # final stage to make prediction
        prediction = h = Basenet.BaseNet.ConvLayer(h,
                                                   shape=(
                                                       3, 3, h.shape.as_list()[-1],
                                                       self.configs.Label_size[2] + 1 + 1),
                                                   # plus one for text region map
                                                   padding=self.configs.padding, strides=(1, 1, 1, 1),
                                                   name='Predict_stage_final')  # ,

        # regularize cos and sin to a squared sum of 1
        cos_sin_sum = tf.sqrt(prediction[:, :, :, 3] * prediction[:, :, :, 3] + prediction[:, :, :, 4] * prediction[:, :, :, 4])
        cos = prediction[:, :, :, 3] / cos_sin_sum
        sin = prediction[:, :, :, 4] / cos_sin_sum
        return tf.stack([prediction[:, :, :, 0], prediction[:, :, :, 1], prediction[:, :, :, 2], cos, sin, prediction[:, :, :, 5], prediction[:, :, :, 6]], axis=-1)
        # TODO at present, the score map is predicted in two channels (1 for pos, 1 for neg), we can also consider sigmoid loss for one channel. but this might be trivial


class network_output_256_down_to_stage_1(network_multi_gpu):
    def _build_network(self):
        with tf.device('/device:cpu:0'):
            self.input_image = tf.placeholder(tf.float32,
                                              shape=(None,
                                                     None, None, 3),
                                              name="InputImage")
            self.Labels = tf.placeholder(tf.float32,
                                         shape=(None,
                                                None, None, 5),
                                         name="Labels")
            self.LossWeights = tf.placeholder(tf.float32,
                                              shape=(None,
                                                     None, None, 1),
                                              name="LossWeights")
            self.resized_Labels = tf.concat([Basenet.BaseNet.Pooling(self.Labels[:, :, :, 0:1], 'max', name='input_resize_1'),
                                             Basenet.BaseNet.Pooling(self.Labels[:, :, :, 1:4], 'avg', name='input_resize_2'),
                                             Basenet.BaseNet.Pooling(self.Labels[:, :, :, 4:5], 'max', name='input_resize_3')],
                                            axis=-1)

            input_split = tf.split(self.input_image, len(self.configs.gpu_list), axis=0)
            Labels_split = tf.split(self.resized_Labels, len(self.configs.gpu_list), axis=0)
            self.global_step = tf.get_variable('global_step', initializer=tf.constant(0, dtype=tf.int32), trainable=False)
            grads_per_gpu = []
            reuse_variables = None
            opt = self._train()
            self.prediction = []
            for i, gpu_id in enumerate(self.configs.gpu_list):
                self.logs['info']('building network for gpu:%s' % gpu_id)
                with tf.device('/device:GPU:%s' % i):
                    with tf.name_scope('model_%s' % gpu_id) as scope:
                        # if True:
                        input_per_gpu = input_split[i]
                        labels_per_gpu = Labels_split[i]
                        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
                            pipe = self._build_back_bone(input_per_gpu)
                            prediction = self._add_prediction_block(pipe)
                            self.prediction.append(prediction)
                            self.total_loss, score_loss, geo_loss, geo_attr, TR_score_loss = self._build_loss(labels_per_gpu, prediction)
                            grads = opt.compute_gradients(self.total_loss)
                            self.logs['info']('prediction layer / loss computation finished')
                            grads_per_gpu.append(grads)
                        if reuse_variables is None:
                            TCL_Loss = tf.summary.scalar("Text_center_line_map_Loss", score_loss)
                            TR_Loss = tf.summary.scalar("Text_Region_map_Loss", TR_score_loss)
                            Loss = tf.summary.scalar("total_loss_Loss", self.total_loss)
                            tf.summary.histogram('cos_his', self.prediction[0][0:1, :, :, 3:4])
                            tf.summary.histogram('cos_his_GT', self.resized_Labels[0:1, :, :, 2:3])
                            tf.summary.histogram('sin_his', self.prediction[0][0:1, :, :, 4:5])
                            tf.summary.histogram('sin_his_GT', self.resized_Labels[0:1, :, :, 3:4])

                            GEO_LOSS = []

                            for i in range(len(geo_attr)):
                                GEO_LOSS.append(tf.summary.scalar('%s_loss' % geo_attr[i], geo_loss[i]))

                            self.summary_op = tf.summary.merge(
                                [TCL_Loss, TR_Loss, Loss] + GEO_LOSS)

                            tf.summary.scalar("Train_Loss", self.total_loss)

                            # TR/TCL map
                            self.TR_TCL_GT_map = tf.concat([self.resized_Labels[:, :, :, 4:5],  # TR
                                                            self.resized_Labels[:, :, :, 0:1],  # TCL
                                                            self.resized_Labels[:, :, :, 4:5] * 0], axis=-1)  # TODO: add new predicting layers: start&end point of text lines
                            L_sum = tf.summary.image('TR_TCL_GT_map', self.TR_TCL_GT_map[0:1, :, :, :], 1)
                            # TR/TCL predicted map
                            self.TR_TCL_Predictied_map = tf.concat([tf.nn.softmax(logits=self.prediction[0][:, :, :, 5:7], dim=-1)[:, :, :, 1:2],
                                                                    tf.cast(self.prediction[0][:, :, :, 6:7] >
                                                                            self.prediction[0][:, :, :, 5:6], tf.float32) *
                                                                    (tf.nn.softmax(logits=self.prediction[0][:, :, :, 0:2], dim=-1)[:, :, :, 1:2]),
                                                                    self.prediction[0][:, :, :, 5:6] * 0],
                                                                   axis=-1)
                            P_sum = tf.summary.image('TR_TCL_Predictied_map', self.TR_TCL_Predictied_map[0:1, :, :, :], 1)
                            # R map
                            max_light = tf.reduce_max(self.resized_Labels[:, :, :, 1:2])
                            self.L_R_map = tf.concat([self.resized_Labels[:, :, :, 1:2],
                                                      self.resized_Labels[:, :, :, 1:2],
                                                      self.resized_Labels[:, :, :, 1:2]], axis=-1) / max_light * 200 / 255
                            self.P_R_map = tf.concat([self.prediction[0][:, :, :, 2:3],
                                                      self.prediction[0][:, :, :, 2:3],
                                                      self.prediction[0][:, :, :, 2:3]], axis=-1) / max_light * 200 / 255

                            L_R_sum = tf.summary.image('Radius_L_map', self.L_R_map[0:1, :, :, :], 1)
                            P_R_sum = tf.summary.image('Radius_P_map', self.P_R_map[0:1, :, :, :], 1)

                            # sin map
                            self.L_sin = tf.concat([(self.resized_Labels[:, :, :, 3:4] + 1) / 2,
                                                    (self.resized_Labels[:, :, :, 3:4] + 1) / 2,
                                                    (self.resized_Labels[:, :, :, 3:4] + 1) / 2], axis=-1)
                            self.P_sin = tf.concat([(self.prediction[0][:, :, :, 4:5] + 1) / 2,
                                                    (self.prediction[0][:, :, :, 4:5] + 1) / 2,
                                                    (self.prediction[0][:, :, :, 4:5] + 1) / 2], axis=-1)
                            L_sin_sum = tf.summary.image('sin_L_map', self.L_sin, 1)
                            P_sin_sum = tf.summary.image('sin_P_map', self.P_sin, 1)

                            # cos map
                            self.L_cos = tf.concat([self.resized_Labels[:, :, :, 2:3],
                                                    self.resized_Labels[:, :, :, 2:3],
                                                    self.resized_Labels[:, :, :, 2:3]], axis=-1)
                            self.P_cos = tf.concat([self.prediction[0][:, :, :, 3:4],
                                                    self.prediction[0][:, :, :, 3:4],
                                                    self.prediction[0][:, :, :, 3:4]], axis=-1)
                            L_cos_sum = tf.summary.image('cos_L_map', self.L_cos[0:1, :, :, :], 1)
                            P_cos_sum = tf.summary.image('cos_P_map', self.P_cos[0:1, :, :, :], 1)

                            im_sum = tf.summary.image('x_original_image', self.input_image[0:1, :, :, ::-1],
                                                      1)  # printing only the first pic

                            self.image_summary_op = tf.summary.merge(
                                [L_sum, P_sum, im_sum])
                            self.image_attributes_summary_op = tf.summary.merge(
                                [L_R_sum, P_R_sum, L_sin_sum, P_sin_sum, L_cos_sum, P_cos_sum])
                        reuse_variables = True

            grads = average_gradients(grads_per_gpu)
            if self.configs.moving_average_decay < 1:
                variable_averages = tf.train.ExponentialMovingAverage(
                    self.configs.moving_average_decay,
                    self.global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)
                with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
                    self.train_step = tf.no_op(name='train_op')
            else:
                self.train_step = opt.apply_gradients(grads, global_step=self.global_step)
            self.prediction = tf.concat(self.prediction, axis=0)

    def _add_prediction_block(self, pipe):
        # prediciton stage
        params = self.configs.US_Params.split(' ')
        activation_functions = {'None': None, 'ReLU': tf.nn.relu, 'cReLU': tf.nn.crelu, 'softmax': tf.nn.softmax}
        if self.configs.upsampling == 'DeConv':
            kwargs = {'output_channel': None,
                      'kernel_size': (int(params[0]), int(params[1])),
                      'strides': (int(params[2]), int(params[3])),
                      'padding': params[4],
                      'activation': activation_functions[params[5]],
                      'name': None}
        else:
            kwargs = {
                'size': int(params)
            }
        pipe['prediction'] = {}
        # stage 1
        pipe['prediction']['stage1'] = h = pipe['output_pipe']['stage5']

        # stage 2-4
        for stage in range(2, self.configs.Predict_stage + 1 + 1):  # [2,5]
            kwargs['output_channel'] = h.shape.as_list()[-1] // 2
            kwargs['name'] = 'stage%d_US' % stage
            h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
            h = tf.concat([h, pipe['output_pipe']['stage' + str(6 - stage)]], axis=-1)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              1, 1, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_1')
            h = tf.nn.relu(h)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              3, 3, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_2')

            pipe['prediction']['stage' + str(stage)] = h = tf.nn.relu(h)

        # penultimate stage
        h = Basenet.BaseNet.ConvLayer(h,
                                      shape=(
                                          3, 3, h.shape.as_list()[-1],
                                          self.configs.predict_channels[self.configs.Predict_stage - 1]),
                                      padding=self.configs.padding, strides=(1, 1, 1, 1),
                                      name='Predict_stage_penul')
        h = tf.nn.relu(h)

        # final stage to make prediction
        prediction = h = Basenet.BaseNet.ConvLayer(h,
                                                   shape=(
                                                       3, 3, h.shape.as_list()[-1],  # TODO kernel size should be (1,1) here(?)
                                                       self.configs.Label_size[2] + 1 + 1),
                                                   # plus one for text region map
                                                   padding=self.configs.padding, strides=(1, 1, 1, 1),
                                                   name='Predict_stage_final')  # ,

        # regularize cos and sin to a squared sum of 1
        cos_sin_sum = tf.sqrt(prediction[:, :, :, 3] * prediction[:, :, :, 3] + prediction[:, :, :, 4] * prediction[:, :, :, 4])
        cos = prediction[:, :, :, 3] / cos_sin_sum
        sin = prediction[:, :, :, 4] / cos_sin_sum
        return tf.stack([prediction[:, :, :, 0], prediction[:, :, :, 1], prediction[:, :, :, 2], cos, sin, prediction[:, :, :, 5], prediction[:, :, :, 6]], axis=-1)
        # TODO at present, the score map is predicted in two channels (1 for pos, 1 for neg), we can also consider sigmoid loss for one channel. but this might be trivial


class network_pixellink_style_simplified_prediction_module(network_multi_gpu):
    def _build_network(self):
        with tf.device('/device:cpu:0'):
            self.input_image = tf.placeholder(tf.float32,
                                              shape=(None,
                                                     None, None, 3),
                                              name="InputImage")
            self.Labels = tf.placeholder(tf.float32,
                                         shape=(None,
                                                None, None, 5),
                                         name="Labels")
            self.LossWeights = tf.placeholder(tf.float32,
                                              shape=(None,
                                                     None, None, 1),
                                              name="LossWeights")
            self.resized_Labels = tf.concat([Basenet.BaseNet.Pooling(self.Labels[:, :, :, 0:1], 'max', name='input_resize_1'),
                                             Basenet.BaseNet.Pooling(self.Labels[:, :, :, 1:4], 'avg', name='input_resize_2'),
                                             Basenet.BaseNet.Pooling(self.Labels[:, :, :, 4:5], 'max', name='input_resize_3')],
                                            axis=-1)
            input_split = tf.split(self.input_image, len(self.configs.gpu_list), axis=0)
            Labels_split = tf.split(self.resized_Labels, len(self.configs.gpu_list), axis=0)
            self.global_step = tf.get_variable('global_step', initializer=tf.constant(0, dtype=tf.int32), trainable=False)
            grads_per_gpu = []
            reuse_variables = None
            opt = self._train()
            self.prediction = []
            for i, gpu_id in enumerate(self.configs.gpu_list):
                self.logs['info']('building network for gpu:%s' % gpu_id)
                with tf.device('/device:GPU:%s' % i):
                    with tf.name_scope('model_%s' % gpu_id) as scope:
                        # if True:
                        input_per_gpu = input_split[i]
                        labels_per_gpu = Labels_split[i]
                        with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
                            pipe = self._build_back_bone(input_per_gpu)
                            prediction = self._add_prediction_block(pipe)
                            self.prediction.append(prediction)
                            self.total_loss, score_loss, geo_loss, geo_attr, TR_score_loss = self._build_loss(labels_per_gpu, prediction)
                            grads = opt.compute_gradients(self.total_loss)
                            self.logs['info']('prediction layer / loss computation finished')
                            grads_per_gpu.append(grads)
                        if reuse_variables is None:
                            TCL_Loss = tf.summary.scalar("Text_center_line_map_Loss", score_loss)
                            TR_Loss = tf.summary.scalar("Text_Region_map_Loss", TR_score_loss)
                            Loss = tf.summary.scalar("total_loss_Loss", self.total_loss)
                            tf.summary.histogram('cos_his', self.prediction[0][0:1, :, :, 3:4])
                            tf.summary.histogram('cos_his_GT', self.resized_Labels[0:1, :, :, 2:3])
                            tf.summary.histogram('sin_his', self.prediction[0][0:1, :, :, 4:5])
                            tf.summary.histogram('sin_his_GT', self.resized_Labels[0:1, :, :, 3:4])

                            GEO_LOSS = []

                            for i in range(len(geo_attr)):
                                GEO_LOSS.append(tf.summary.scalar('%s_loss' % geo_attr[i], geo_loss[i]))

                            self.summary_op = tf.summary.merge(
                                [TCL_Loss, TR_Loss, Loss] + GEO_LOSS)

                            tf.summary.scalar("Train_Loss", self.total_loss)

                            # TR/TCL map
                            self.TR_TCL_GT_map = tf.concat([self.resized_Labels[:, :, :, 4:5],  # TR
                                                            self.resized_Labels[:, :, :, 0:1],  # TCL
                                                            self.resized_Labels[:, :, :, 4:5] * 0], axis=-1)  # TODO: add new predicting layers: start&end point of text lines
                            L_sum = tf.summary.image('TR_TCL_GT_map', self.TR_TCL_GT_map[0:1, :, :, :], 1)
                            # TR/TCL predicted map
                            self.TR_TCL_Predictied_map = tf.concat([tf.nn.softmax(logits=self.prediction[0][:, :, :, 5:7], dim=-1)[:, :, :, 1:2],
                                                                    tf.cast(self.prediction[0][:, :, :, 6:7] >
                                                                            self.prediction[0][:, :, :, 5:6], tf.float32) *
                                                                    (tf.nn.softmax(logits=self.prediction[0][:, :, :, 0:2], dim=-1)[:, :, :, 1:2]),
                                                                    self.prediction[0][:, :, :, 5:6] * 0],
                                                                   axis=-1)
                            P_sum = tf.summary.image('TR_TCL_Predictied_map', self.TR_TCL_Predictied_map[0:1, :, :, :], 1)
                            # R map
                            max_light = tf.reduce_max(self.resized_Labels[:, :, :, 1:2])
                            self.L_R_map = tf.concat([self.resized_Labels[:, :, :, 1:2],
                                                      self.resized_Labels[:, :, :, 1:2],
                                                      self.resized_Labels[:, :, :, 1:2]], axis=-1) / max_light * 200 / 255
                            self.P_R_map = tf.concat([self.prediction[0][:, :, :, 2:3],
                                                      self.prediction[0][:, :, :, 2:3],
                                                      self.prediction[0][:, :, :, 2:3]], axis=-1) / max_light * 200 / 255

                            L_R_sum = tf.summary.image('Radius_L_map', self.L_R_map[0:1, :, :, :], 1)
                            P_R_sum = tf.summary.image('Radius_P_map', self.P_R_map[0:1, :, :, :], 1)

                            # sin map
                            self.L_sin = tf.concat([(self.resized_Labels[:, :, :, 3:4] + 1) / 2,
                                                    (self.resized_Labels[:, :, :, 3:4] + 1) / 2,
                                                    (self.resized_Labels[:, :, :, 3:4] + 1) / 2], axis=-1)
                            self.P_sin = tf.concat([(self.prediction[0][:, :, :, 4:5] + 1) / 2,
                                                    (self.prediction[0][:, :, :, 4:5] + 1) / 2,
                                                    (self.prediction[0][:, :, :, 4:5] + 1) / 2], axis=-1)
                            L_sin_sum = tf.summary.image('sin_L_map', self.L_sin, 1)
                            P_sin_sum = tf.summary.image('sin_P_map', self.P_sin, 1)

                            # cos map
                            self.L_cos = tf.concat([self.resized_Labels[:, :, :, 2:3],
                                                    self.resized_Labels[:, :, :, 2:3],
                                                    self.resized_Labels[:, :, :, 2:3]], axis=-1)
                            self.P_cos = tf.concat([self.prediction[0][:, :, :, 3:4],
                                                    self.prediction[0][:, :, :, 3:4],
                                                    self.prediction[0][:, :, :, 3:4]], axis=-1)
                            L_cos_sum = tf.summary.image('cos_L_map', self.L_cos[0:1, :, :, :], 1)
                            P_cos_sum = tf.summary.image('cos_P_map', self.P_cos[0:1, :, :, :], 1)

                            im_sum = tf.summary.image('x_original_image', self.input_image[0:1, :, :, ::-1],
                                                      1)  # printing only the first pic

                            self.image_summary_op = tf.summary.merge(
                                [L_sum, P_sum, im_sum])
                            self.image_attributes_summary_op = tf.summary.merge(
                                [L_R_sum, P_R_sum, L_sin_sum, P_sin_sum, L_cos_sum, P_cos_sum])
                        reuse_variables = True

            grads = average_gradients(grads_per_gpu)
            if self.configs.moving_average_decay < 1:
                variable_averages = tf.train.ExponentialMovingAverage(
                    self.configs.moving_average_decay,
                    self.global_step)
                variables_averages_op = variable_averages.apply(tf.trainable_variables())
                apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)
                with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
                    self.train_step = tf.no_op(name='train_op')
            else:
                self.train_step = opt.apply_gradients(grads, global_step=self.global_step)
            self.prediction = tf.concat(self.prediction, axis=0)

    def _add_prediction_block(self, pipe):
        # prediciton stage
        params = self.configs.US_Params.split(' ')
        activation_functions = {'None': None, 'ReLU': tf.nn.relu, 'cReLU': tf.nn.crelu, 'softmax': tf.nn.softmax}
        if self.configs.upsampling == 'DeConv':
            kwargs = {'output_channel': None,
                      'kernel_size': (int(params[0]), int(params[1])),
                      'strides': (int(params[2]), int(params[3])),
                      'padding': params[4],
                      'activation': activation_functions[params[5]],
                      'name': None}
        else:
            kwargs = {
                'size': int(params)
            }
        pipe['prediction'] = {}
        # stage 1
        h = pipe['output_pipe']['stage5']
        h = tf.nn.relu(Basenet.BaseNet.ConvLayer(h,
                                                 shape=(
                                                     3, 3,
                                                     h.shape.as_list()[-1],
                                                     h.shape.as_list()[-1]),
                                                 padding=self.configs.padding, strides=(1, 1, 1, 1),
                                                 name='fc6'))
        h = tf.nn.relu(Basenet.BaseNet.ConvLayer(h,
                                                 shape=(
                                                     3, 3,
                                                     h.shape.as_list()[-1],
                                                     h.shape.as_list()[-1]),
                                                 padding=self.configs.padding, strides=(1, 1, 1, 1),
                                                 name='fc7'))
        h = (Basenet.BaseNet.ConvLayer(h,
                                       shape=(
                                           1, 1,
                                           h.shape.as_list()[-1],
                                           7),
                                       padding=self.configs.padding, strides=(1, 1, 1, 1),
                                       name='stage_1_conv'))
        kwargs['output_channel'] = 7
        kwargs['name'] = 'stage%d_US' % 1
        pipe['prediction']['stage1'] = h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)

        # stage 2-4
        for stage in range(2, self.configs.Predict_stage + 1 + 1):  # [2,5]
            h += tf.nn.relu(Basenet.BaseNet.ConvLayer(pipe['output_pipe']['stage' + str(6 - stage)],
                                                      shape=(
                1, 1,
                pipe['output_pipe']['stage' + str(6 - stage)].shape.as_list()[-1],
                7),
                padding=self.configs.padding, strides=(1, 1, 1, 1),
                name='stage_%d_conv' % stage))
            if stage < self.configs.Predict_stage + 1:
                kwargs['name'] = 'stage%d_US' % stage
                h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
            pipe['prediction']['stage' + str(stage)] = h

        # final stage
        h = Basenet.BaseNet.ConvLayer(h,
                                      shape=(
                                          1, 1, h.shape.as_list()[-1],
                                          self.configs.Label_size[2] + 1 + 1),
                                      padding=self.configs.padding, strides=(1, 1, 1, 1),
                                      name='Predict_stage_final')
        prediction = h

        # regularize cos and sin to a squared sum of 1
        cos_sin_sum = tf.sqrt(prediction[:, :, :, 3] * prediction[:, :, :, 3] + prediction[:, :, :, 4] * prediction[:, :, :, 4])
        cos = prediction[:, :, :, 3] / cos_sin_sum
        sin = prediction[:, :, :, 4] / cos_sin_sum
        return tf.stack([prediction[:, :, :, 0], prediction[:, :, :, 1], prediction[:, :, :, 2], cos, sin, prediction[:, :, :, 5], prediction[:, :, :, 6]], axis=-1)
        # TODO at present, the score map is predicted in two channels (1 for pos, 1 for neg), we can also consider sigmoid loss for one channel. but this might be trivial


class network_TCL(network_multi_gpu):
    def _add_prediction_block(self, pipe):
        # prediciton stage
        params = self.configs.US_Params.split(' ')
        activation_functions = {'None': None, 'ReLU': tf.nn.relu, 'cReLU': tf.nn.crelu, 'softmax': tf.nn.softmax}
        if self.configs.upsampling == 'DeConv':
            kwargs = {'output_channel': None,
                      'kernel_size': (int(params[0]), int(params[1])),
                      'strides': (int(params[2]), int(params[3])),
                      'padding': params[4],
                      'activation': activation_functions[params[5]],
                      'name': None}
        else:
            kwargs = {
                'size': int(params)
            }
        pipe['prediction'] = {}
        # stage 1
        pipe['prediction']['stage1'] = h = pipe['output_pipe']['stage5']

        # stage 2-4
        for stage in range(2, self.configs.Predict_stage + 1):  # [2,5)
            kwargs['output_channel'] = h.shape.as_list()[-1] // 2
            kwargs['name'] = 'stage%d_US' % stage
            h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
            h = tf.concat([h, pipe['output_pipe']['stage' + str(6 - stage)]], axis=-1)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              1, 1, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_1')
            h = tf.nn.relu(h)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              3, 3, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_2')

            pipe['prediction']['stage' + str(stage)] = h = tf.nn.relu(h)

        # stage 5-6
        kwargs['output_channel'] = self.configs.predict_channels[-1]
        kwargs['name'] = 'stage%d_US' % (-2)
        h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
        kwargs['name'] = 'stage%d_US' % (-1)
        h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
        h = Basenet.BaseNet.ConvLayer(h,
                                      shape=(
                                          3, 3, h.shape.as_list()[-1],
                                          self.configs.predict_channels[self.configs.Predict_stage - 1]),
                                      padding=self.configs.padding, strides=(1, 1, 1, 1),
                                      name='Predict_stage_penul')
        h = tf.nn.relu(h)

        # final stage to make prediction
        prediction = Basenet.BaseNet.ConvLayer(h,
                                               shape=(
                                                   3, 3, h.shape.as_list()[-1],  # TODO kernel size should be (1,1) here(?)
                                                   self.configs.Label_size[2] + 1 + 1),
                                               # plus one for text region map
                                               padding=self.configs.padding, strides=(1, 1, 1, 1),
                                               name='Predict_stage_final')  # ,

        # regularize cos and sin to a squared sum of 1
        cos_sin_sum = tf.sqrt(prediction[:, :, :, 3] * prediction[:, :, :, 3] + prediction[:, :, :, 4] * prediction[:, :, :, 4])
        cos = prediction[:, :, :, 3] / cos_sin_sum
        sin = prediction[:, :, :, 4] / cos_sin_sum

        TCL = Basenet.BaseNet.ConvLayer(tf.concat([h, prediction], axis=-1),
                                        shape=(
            1, 1, h.shape.as_list()[-1] + prediction.shape.as_list()[-1],
            # TODO kernel size should be (1,1) here(?)
            2),
            # plus one for text region map
            padding=self.configs.padding, strides=(1, 1, 1, 1),
            name='TCL')

        return tf.stack([TCL[:, :, :, 0], TCL[:, :, :, 1], prediction[:, :, :, 2], cos, sin, prediction[:, :, :, 5], prediction[:, :, :, 6]], axis=-1)
        # TODO at present, the score map is predicted in two channels (1 for pos, 1 for neg), we can also consider sigmoid loss for one channel. but this might be trivial

    def _build_loss(self, Labels, prediction):
        with tf.name_scope('TR_loss'):
            def pos_mask_TR():
                # from self.Labels[:,:,:,0:1]
                return Labels[:, :, :, 4:5] > 0
            pos_TR = tf.cast(pos_mask_TR(), tf.float32)
            pos_num_TR = tf.reduce_sum(pos_TR) + 1e-3
            neg_num_TR = tf.cast(3 * pos_num_TR + 1, tf.int32)  # in case, OHNM is used
            # TR score map loss
            singel_labels_TR = flatten(Labels[:, :, :, 4:5])
            one_hot_labels_TR = tf.concat([1 - singel_labels_TR, singel_labels_TR], axis=-1)
            loss_TR = tf.losses.softmax_cross_entropy(one_hot_labels_TR,
                                                      flatten(prediction[:, :, :, 5:7]),
                                                      reduction=tf.losses.Reduction.NONE)
            pos_flatten_TR = tf.reshape(flatten(pos_TR), shape=(-1,))

            pos_loss_TR = loss_TR * pos_flatten_TR
            neg_losses_TR = loss_TR * (1 - pos_flatten_TR)
            neg_loss_TR = tf.nn.top_k(neg_losses_TR, k=tf.reduce_min((neg_num_TR, tf.size(neg_losses_TR)))).values
            TR_score_loss = (tf.reduce_sum(pos_loss_TR) / pos_num_TR +
                             tf.reduce_sum(neg_loss_TR) / pos_num_TR)  # top_k in use

        with tf.name_scope('TCL_loss'):
            def pos_mask():
                # from self.Labels[:,:,:,0:1]
                return Labels[:, :, :, 0:1] > 0
            pos = tf.cast(pos_mask(), tf.float32)
            pos_num = tf.reduce_sum(pos) + 1e-3
            neg_num = tf.cast(3 * pos_num + 1, tf.int32)  # in case, OHNM is used
            # score map loss
            singel_labels = flatten(Labels[:, :, :, 0:1])
            one_hot_labels = tf.concat([1 - singel_labels, singel_labels], axis=-1)
            loss = tf.losses.softmax_cross_entropy(one_hot_labels,
                                                   flatten(prediction[:, :, :, 0:2]),
                                                   reduction=tf.losses.Reduction.NONE)

            pos_flatten = tf.reshape(flatten(pos), shape=(-1,))
            pos_loss = loss * pos_flatten
            neg_loss = loss * (1 - pos_flatten)
            total_num = tf.cast(tf.size(pos), tf.float32)
            # neg_loss=tf.nn.top_k(neg_losses,k=neg_num).values
            score_loss = (tf.reduce_sum(pos_loss) / pos_num +
                          tf.reduce_sum(neg_loss) / (total_num - pos_num))  # top_k not in use TODO: rebalance the pos/neg number !

        with tf.name_scope('Geo_loss'):
            geo_attr = ['radius', 'cos', 'sin']
            geo_loss = []
            total_loss = score_loss + TR_score_loss  # for training
            for i in range(4 - 1):  # self.configs.Label_size[2]-1):
                geo_loss.append(smooth_l1_loss(flatten(Labels[:, :, :, i + 1:i + 2] * pos),
                                               flatten(prediction[:, :, :, i + 2:i + 3] * pos)
                                               ) / pos_num)
                total_loss += geo_loss[-1]

        return total_loss, score_loss, geo_loss, geo_attr, TR_score_loss


class network_new_loss(network_multi_gpu):
    def _build_loss(self, Labels, prediction):
        with tf.name_scope('TR_loss'):
            def pos_mask_TR():
                # from self.Labels[:,:,:,0:1]
                return Labels[:, :, :, 4:5] > 0
            pos_TR = tf.cast(pos_mask_TR(), tf.float32)
            pos_num_TR = tf.reduce_sum(pos_TR) + 1e-3
            neg_num_TR = tf.cast(3 * pos_num_TR + 1, tf.int32)  # in case, OHNM is used
            # TR score map loss
            singel_labels_TR = flatten(Labels[:, :, :, 4:5])
            one_hot_labels_TR = tf.concat([1 - singel_labels_TR, singel_labels_TR], axis=-1)
            loss_TR = tf.losses.softmax_cross_entropy(one_hot_labels_TR,
                                                      flatten(prediction[:, :, :, 5:7]),
                                                      reduction=tf.losses.Reduction.NONE)
            pos_flatten_TR = tf.reshape(flatten(pos_TR), shape=(-1,))

            pos_loss_TR = loss_TR * pos_flatten_TR
            neg_losses_TR = loss_TR * (1 - pos_flatten_TR)
            neg_loss_TR = tf.nn.top_k(neg_losses_TR, k=tf.reduce_min((neg_num_TR, tf.size(neg_losses_TR)))).values
            TR_score_loss = (tf.reduce_sum(pos_loss_TR) / pos_num_TR +
                             tf.reduce_sum(neg_loss_TR) / pos_num_TR)  # top_k in use

        with tf.name_scope('TCL_loss'):
            def pos_mask():
                # from self.Labels[:,:,:,0:1]
                return Labels[:, :, :, 0:1] > 0
            pos = tf.cast(pos_mask(), tf.float32)
            pos_num = tf.reduce_sum(pos) + 1e-3
            neg_num = tf.cast(3 * pos_num + 1, tf.int32)  # in case, OHNM is used
            # score map loss
            singel_labels = flatten(Labels[:, :, :, 0:1])
            one_hot_labels = tf.concat([1 - singel_labels, singel_labels], axis=-1)
            loss = tf.losses.softmax_cross_entropy(one_hot_labels,
                                                   flatten(prediction[:, :, :, 0:2]),
                                                   reduction=tf.losses.Reduction.NONE)

            pos_flatten = tf.reshape(flatten(pos), shape=(-1,))
            pos_loss = loss * pos_flatten
            neg_loss = loss * (1 - pos_flatten)
            total_num = tf.cast(tf.size(pos), tf.float32)
            # neg_loss=tf.nn.top_k(neg_losses,k=neg_num).values
            score_loss = (tf.reduce_sum(pos_loss) / pos_num +
                          tf.reduce_sum(neg_loss) / (total_num - pos_num))  # top_k not in use TODO: rebalance the pos/neg number !

        with tf.name_scope('Geo_loss'):
            geo_attr = ['radius', 'cos', 'sin']
            geo_loss = []
            total_loss = score_loss + TR_score_loss  # for training
            for i in range(4 - 1):  # self.configs.Label_size[2]-1):
                geo_loss.append(smooth_l1_loss(flatten(Labels[:, :, :, i + 1:i + 2] * pos),
                                               flatten(prediction[:, :, :, i + 2:i + 3] * pos)
                                               ) / pos_num)
                total_loss += geo_loss[-1]

        return total_loss, score_loss, geo_loss, geo_attr, TR_score_loss


class network_TCL_old_loss(network_multi_gpu):
    def _add_prediction_block(self, pipe):
        # prediciton stage
        params = self.configs.US_Params.split(' ')
        activation_functions = {'None': None, 'ReLU': tf.nn.relu, 'cReLU': tf.nn.crelu, 'softmax': tf.nn.softmax}
        if self.configs.upsampling == 'DeConv':
            kwargs = {'output_channel': None,
                      'kernel_size': (int(params[0]), int(params[1])),
                      'strides': (int(params[2]), int(params[3])),
                      'padding': params[4],
                      'activation': activation_functions[params[5]],
                      'name': None}
        else:
            kwargs = {
                'size': int(params)
            }
        pipe['prediction'] = {}
        # stage 1
        pipe['prediction']['stage1'] = h = pipe['output_pipe']['stage5']

        # stage 2-4
        for stage in range(2, self.configs.Predict_stage + 1):  # [2,5)
            kwargs['output_channel'] = h.shape.as_list()[-1] // 2
            kwargs['name'] = 'stage%d_US' % stage
            h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
            h = tf.concat([h, pipe['output_pipe']['stage' + str(6 - stage)]], axis=-1)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              1, 1, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_1')
            h = tf.nn.relu(h)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              3, 3, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_2')

            pipe['prediction']['stage' + str(stage)] = h = tf.nn.relu(h)

        # stage 5-6
        kwargs['output_channel'] = self.configs.predict_channels[-1]
        kwargs['name'] = 'stage%d_US' % (-2)
        h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
        kwargs['name'] = 'stage%d_US' % (-1)
        h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
        h = Basenet.BaseNet.ConvLayer(h,
                                      shape=(
                                          3, 3, h.shape.as_list()[-1],
                                          self.configs.predict_channels[self.configs.Predict_stage - 1]),
                                      padding=self.configs.padding, strides=(1, 1, 1, 1),
                                      name='Predict_stage_penul')
        h = tf.nn.relu(h)

        # final stage to make prediction
        prediction = Basenet.BaseNet.ConvLayer(h,
                                               shape=(
                                                   3, 3, h.shape.as_list()[-1],  # TODO kernel size should be (1,1) here(?)
                                                   self.configs.Label_size[2] + 1 + 1),
                                               # plus one for text region map
                                               padding=self.configs.padding, strides=(1, 1, 1, 1),
                                               name='Predict_stage_final')  # ,

        # regularize cos and sin to a squared sum of 1
        cos_sin_sum = tf.sqrt(prediction[:, :, :, 3] * prediction[:, :, :, 3] + prediction[:, :, :, 4] * prediction[:, :, :, 4])
        cos = prediction[:, :, :, 3] / cos_sin_sum
        sin = prediction[:, :, :, 4] / cos_sin_sum

        TCL = Basenet.BaseNet.ConvLayer(tf.concat([h, prediction], axis=-1),
                                        shape=(
            1, 1, h.shape.as_list()[-1] + prediction.shape.as_list()[-1],
            # TODO kernel size should be (1,1) here(?)
            2),
            # plus one for text region map
            padding=self.configs.padding, strides=(1, 1, 1, 1),
            name='TCL')

        return tf.stack([TCL[:, :, :, 0], TCL[:, :, :, 1], prediction[:, :, :, 2], cos, sin, prediction[:, :, :, 5], prediction[:, :, :, 6]], axis=-1)
        # TODO at present, the score map is predicted in two channels (1 for pos, 1 for neg), we can also consider sigmoid loss for one channel. but this might be trivial


class network_output_down_to_stage_1_deconv_1_kernel_1(network_multi_gpu):
    """
    This model is the best performing by now.
    """
    # this function is to build blocks for predictions of pixels

    def _add_prediction_block(self, pipe):
        # prediciton stage
        params = self.configs.US_Params.split(' ')
        activation_functions = {'None': None, 'ReLU': tf.nn.relu, 'cReLU': tf.nn.crelu, 'softmax': tf.nn.softmax}
        if self.configs.upsampling == 'DeConv':
            kwargs = {'output_channel': None,
                      'kernel_size': (int(params[0]), int(params[1])),
                      'strides': (int(params[2]), int(params[3])),
                      'padding': params[4],
                      'activation': activation_functions[params[5]],
                      'name': None}
        else:
            kwargs = {
                'size': int(params)
            }
        pipe['prediction'] = {}
        # stage 1
        pipe['prediction']['stage1'] = h = pipe['output_pipe']['stage5']

        # stage 2-4
        for stage in range(2, self.configs.Predict_stage + 1 + 1):  # [2,5)
            kwargs['output_channel'] = h.shape.as_list()[-1] // 2
            kwargs['name'] = 'stage%d_US' % stage
            h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
            h = tf.concat([h, pipe['output_pipe']['stage' + str(6 - stage)]], axis=-1)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              1, 1, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_1')
            h = tf.nn.relu(h)
            h = Basenet.BaseNet.ConvLayer(h,
                                          shape=(
                                              3, 3, h.shape.as_list()[-1],
                                              self.configs.predict_channels[stage - 2]),
                                          padding=self.configs.padding, strides=(1, 1, 1, 1),
                                          name='Predict_stage_' + str(stage) + '_Conv_2')

            pipe['prediction']['stage' + str(stage)] = h = tf.nn.relu(h)

        # stage 5-6
        kwargs['output_channel'] = self.configs.predict_channels[-1]
        kwargs['name'] = 'stage%d_US' % (-2)
        h = Basenet.BaseNet.upsampling(h, upsampling=self.configs.upsampling, **kwargs)
        h = Basenet.BaseNet.ConvLayer(h,
                                      shape=(
                                          3, 3, h.shape.as_list()[-1],
                                          self.configs.predict_channels[self.configs.Predict_stage - 1]),
                                      padding=self.configs.padding, strides=(1, 1, 1, 1),
                                      name='Predict_stage_penul')
        h = tf.nn.relu(h)

        # final stage to make prediction
        prediction = h = Basenet.BaseNet.ConvLayer(h,
                                                   shape=(
                                                       1, 1, h.shape.as_list()[-1],
                                                       self.configs.Label_size[2] + 1 + 1),
                                                   # plus one for text region map
                                                   padding=self.configs.padding, strides=(1, 1, 1, 1),
                                                   name='Predict_stage_final')  # ,

        # regularize cos and sin to a squared sum of 1
        cos_sin_sum = tf.sqrt(prediction[:, :, :, 3] * prediction[:, :, :, 3] + prediction[:, :, :, 4] * prediction[:, :, :, 4])
        cos = prediction[:, :, :, 3] / cos_sin_sum
        sin = prediction[:, :, :, 4] / cos_sin_sum
        return tf.stack([prediction[:, :, :, 0], prediction[:, :, :, 1], prediction[:, :, :, 2], cos, sin, prediction[:, :, :, 5], prediction[:, :, :, 6]], axis=-1)
        # TODO at present, the score map is predicted in two channels (1 for pos, 1 for neg), we can also consider sigmoid loss for one channel. but this might be trivial


class network_geo_balanced(network_output_down_to_stage_1_deconv_1_kernel_1):
    def _build_loss(self, Labels, prediction):
        with tf.name_scope('TR_loss'):
            def pos_mask_TR():
                # from self.Labels[:,:,:,0:1]
                return Labels[:, :, :, 4:5] > 0
            pos_TR = tf.cast(pos_mask_TR(), tf.float32)
            pos_num_TR = tf.reduce_sum(pos_TR) + 1e-3
            neg_num_TR = tf.cast(3 * pos_num_TR + 1, tf.int32)  # in case, OHNM is used
            # TR score map loss
            singel_labels_TR = flatten(Labels[:, :, :, 4:5])
            one_hot_labels_TR = tf.concat([1 - singel_labels_TR, singel_labels_TR], axis=-1)
            loss_TR = tf.losses.softmax_cross_entropy(one_hot_labels_TR,
                                                      flatten(prediction[:, :, :, 5:7]),
                                                      reduction=tf.losses.Reduction.NONE)
            pos_flatten_TR = tf.reshape(flatten(pos_TR), shape=(-1,))

            pos_loss_TR = loss_TR * pos_flatten_TR
            neg_losses_TR = loss_TR * (1 - pos_flatten_TR)
            neg_loss_TR = tf.nn.top_k(neg_losses_TR, k=tf.reduce_min((neg_num_TR, tf.size(neg_losses_TR)))).values
            TR_score_loss = (tf.reduce_sum(pos_loss_TR) / pos_num_TR +
                             tf.reduce_sum(neg_loss_TR) / pos_num_TR)  # top_k in use

        with tf.name_scope('TCL_loss'):
            def pos_mask():
                # from self.Labels[:,:,:,0:1]
                return Labels[:, :, :, 0:1] > 0
            pos = tf.cast(pos_mask(), tf.float32)
            pos_num = tf.reduce_sum(pos) + 1e-3
            neg_num = tf.cast(3 * pos_num + 1, tf.int32)  # in case, OHNM is used
            # score map loss
            singel_labels = flatten(Labels[:, :, :, 0:1])
            one_hot_labels = tf.concat([1 - singel_labels, singel_labels], axis=-1)
            loss = tf.losses.softmax_cross_entropy(one_hot_labels,
                                                   flatten(prediction[:, :, :, 0:2]),
                                                   reduction=tf.losses.Reduction.NONE)

            pos_flatten = tf.reshape(flatten(pos), shape=(-1,))
            pos_loss = loss * pos_flatten * pos_flatten_TR
            neg_loss = loss * (1 - pos_flatten) * pos_flatten_TR
            # neg_loss=tf.nn.top_k(neg_losses,k=neg_num).values
            score_loss = (tf.reduce_sum(pos_loss) / pos_num +
                          tf.reduce_sum(neg_loss) / pos_num)  # top_k not in use TODO: rebalance the pos/neg number !

        with tf.name_scope('Geo_loss'):
            geo_attr = ['radius', 'cos', 'sin']
            geo_loss = []
            total_loss = score_loss + TR_score_loss  # for training
            for i in range(4 - 1):  # self.configs.Label_size[2]-1):
                if i == 0:
                    label = flatten(Labels[:, :, :, i + 1:i + 2] * pos) + 1e-5
                    pred = flatten(prediction[:, :, :, i + 2:i + 3] * pos) + 1e-5
                    geo_loss.append(smooth_l1_loss(pred / label,
                                                   label * 0 + 1
                                                   ) / pos_num)
                    total_loss += geo_loss[-1]
                else:
                    geo_loss.append(smooth_l1_loss(flatten(Labels[:, :, :, i + 1:i + 2] * pos),
                                                   flatten(prediction[:, :, :, i + 2:i + 3] * pos)
                                                   ) / pos_num)
                    total_loss += geo_loss[-1]

        return total_loss, score_loss, geo_loss, geo_attr, TR_score_loss


class network_IB(network_output_down_to_stage_1_deconv_1_kernel_1):
    """
    use balanced crossentropy loss for TR/TCL
    loss for TCL only considers those in TR area
    """

    def _build_loss(self, Labels, prediction):
        with tf.name_scope('TR_loss'):
            def pos_mask_TR():
                # from self.Labels[:,:,:,0:1]
                return Labels[:, :, :, 4:5] > 0
            pos_TR = tf.cast(pos_mask_TR(), tf.float32)
            pos_num_TR = tf.reduce_sum(pos_TR) + 1e-3
            neg_num_TR = tf.cast(tf.size(pos_TR), tf.float32) - pos_num_TR
            # TR score map loss
            singel_labels_TR = flatten(Labels[:, :, :, 4:5])
            one_hot_labels_TR = tf.concat([1 - singel_labels_TR, singel_labels_TR], axis=-1)
            loss_TR = tf.losses.softmax_cross_entropy(one_hot_labels_TR,
                                                      flatten(prediction[:, :, :, 5:7]),
                                                      reduction=tf.losses.Reduction.NONE)
            pos_flatten_TR = tf.reshape(flatten(pos_TR), shape=(-1,))

            pos_loss_TR = loss_TR * pos_flatten_TR
            neg_loss_TR = loss_TR * (1 - pos_flatten_TR)
            TR_score_loss = (tf.reduce_sum(pos_loss_TR) / pos_num_TR +
                             tf.reduce_sum(neg_loss_TR) / neg_num_TR)  # top_k in use

        with tf.name_scope('TCL_loss'):
            def pos_mask():
                # from self.Labels[:,:,:,0:1]
                return Labels[:, :, :, 0:1] > 0
            pos = tf.cast(pos_mask(), tf.float32)
            pos_num = tf.reduce_sum(pos) + 1e-3
            neg_num = pos_num_TR - pos_num
            # score map loss
            singel_labels = flatten(Labels[:, :, :, 0:1])
            one_hot_labels = tf.concat([1 - singel_labels, singel_labels], axis=-1)
            loss = tf.losses.softmax_cross_entropy(one_hot_labels,
                                                   flatten(prediction[:, :, :, 0:2]),
                                                   reduction=tf.losses.Reduction.NONE)

            pos_flatten = tf.reshape(flatten(pos), shape=(-1,))
            pos_loss = loss * pos_flatten * pos_flatten_TR
            neg_loss = loss * (1 - pos_flatten) * pos_flatten_TR
            # neg_loss=tf.nn.top_k(neg_losses,k=neg_num).values
            score_loss = (tf.reduce_sum(pos_loss) / pos_num +
                          tf.reduce_sum(neg_loss) / neg_num)  # top_k not in use TODO: rebalance the pos/neg number !

        with tf.name_scope('Geo_loss'):
            geo_attr = ['radius', 'cos', 'sin']
            geo_loss = []
            total_loss = score_loss + TR_score_loss  # for training
            for i in range(4 - 1):  # self.configs.Label_size[2]-1):
                geo_loss.append(smooth_l1_loss(flatten(Labels[:, :, :, i + 1:i + 2] * pos),
                                               flatten(prediction[:, :, :, i + 2:i + 3] * pos)
                                               ) / pos_num)
                total_loss += geo_loss[-1]

        return total_loss, score_loss, geo_loss, geo_attr, TR_score_loss


class network_IB_TCL(network_output_down_to_stage_1_deconv_1_kernel_1):
    """
    use balanced crossentropy loss for TR/TCL
    loss for TCL considers all positions
    """

    def _build_loss(self, Labels, prediction):
        with tf.name_scope('TR_loss'):
            def pos_mask_TR():
                # from self.Labels[:,:,:,0:1]
                return Labels[:, :, :, 4:5] > 0
            pos_TR = tf.cast(pos_mask_TR(), tf.float32)
            pos_num_TR = tf.reduce_sum(pos_TR) + 1e-3
            neg_num_TR = tf.cast(tf.size(pos_TR), tf.float32) - pos_num_TR
            # TR score map loss
            singel_labels_TR = flatten(Labels[:, :, :, 4:5])
            one_hot_labels_TR = tf.concat([1 - singel_labels_TR, singel_labels_TR], axis=-1)
            loss_TR = tf.losses.softmax_cross_entropy(one_hot_labels_TR,
                                                      flatten(prediction[:, :, :, 5:7]),
                                                      reduction=tf.losses.Reduction.NONE)
            pos_flatten_TR = tf.reshape(flatten(pos_TR), shape=(-1,))

            pos_loss_TR = loss_TR * pos_flatten_TR
            neg_loss_TR = loss_TR * (1 - pos_flatten_TR)
            TR_score_loss = (tf.reduce_sum(pos_loss_TR) / pos_num_TR +
                             tf.reduce_sum(neg_loss_TR) / neg_num_TR)  # top_k in use

        with tf.name_scope('TCL_loss'):
            def pos_mask():
                # from self.Labels[:,:,:,0:1]
                return Labels[:, :, :, 0:1] > 0
            pos = tf.cast(pos_mask(), tf.float32)
            pos_num = tf.reduce_sum(pos) + 1e-3
            neg_num = tf.cast(tf.size(pos_TR), tf.float32) - pos_num
            # score map loss
            singel_labels = flatten(Labels[:, :, :, 0:1])
            one_hot_labels = tf.concat([1 - singel_labels, singel_labels], axis=-1)
            loss = tf.losses.softmax_cross_entropy(one_hot_labels,
                                                   flatten(prediction[:, :, :, 0:2]),
                                                   reduction=tf.losses.Reduction.NONE)

            pos_flatten = tf.reshape(flatten(pos), shape=(-1,))
            pos_loss = loss * pos_flatten
            neg_loss = loss * (1 - pos_flatten)
            # neg_loss=tf.nn.top_k(neg_losses,k=neg_num).values
            score_loss = (tf.reduce_sum(pos_loss) / pos_num +
                          tf.reduce_sum(neg_loss) / neg_num)  # top_k not in use TODO: rebalance the pos/neg number !

        with tf.name_scope('Geo_loss'):
            geo_attr = ['radius', 'cos', 'sin']
            geo_loss = []
            total_loss = score_loss + TR_score_loss  # for training
            for i in range(4 - 1):  # self.configs.Label_size[2]-1):
                geo_loss.append(smooth_l1_loss(flatten(Labels[:, :, :, i + 1:i + 2] * pos),
                                               flatten(prediction[:, :, :, i + 2:i + 3] * pos)
                                               ) / pos_num)
                total_loss += geo_loss[-1]

        return total_loss, score_loss, geo_loss, geo_attr, TR_score_loss


class network_IAB(network_output_down_to_stage_1_deconv_1_kernel_1):
    """
    use balanced crossentropy loss for TR/TCL
    loss for TCL considers all positions
    """

    def _build_loss(self, Labels, prediction):

        Loss_weights = flatten(self.LossWeights[:, :, :, :])

        with tf.name_scope('TR_loss'):
            def pos_mask_TR():
                # from self.Labels[:,:,:,0:1]
                return Labels[:, :, :, 4:5] > 0
            pos_TR = tf.cast(pos_mask_TR(), tf.float32)
            pos_num_TR = tf.reduce_sum(pos_TR) + 1e-3
            neg_num_TR = tf.cast(tf.size(pos_TR), tf.float32) - pos_num_TR
            # TR score map loss
            singel_labels_TR = flatten(Labels[:, :, :, 4:5])
            one_hot_labels_TR = tf.concat([1 - singel_labels_TR, singel_labels_TR], axis=-1)
            loss_TR = tf.losses.softmax_cross_entropy(one_hot_labels_TR,
                                                      flatten(prediction[:, :, :, 5:7]),
                                                      reduction=tf.losses.Reduction.NONE)
            pos_flatten_TR = tf.reshape(flatten(pos_TR), shape=(-1,))

            pos_loss_TR = loss_TR * pos_flatten_TR * Loss_weights
            neg_loss_TR = loss_TR * (1 - pos_flatten_TR) * Loss_weights
            TR_score_loss = (tf.reduce_sum(pos_loss_TR) / pos_num_TR +
                             tf.reduce_sum(neg_loss_TR) / neg_num_TR)  # top_k in use

        with tf.name_scope('TCL_loss'):
            def pos_mask():
                # from self.Labels[:,:,:,0:1]
                return Labels[:, :, :, 0:1] > 0
            pos = tf.cast(pos_mask(), tf.float32)
            pos_num = tf.reduce_sum(pos) + 1e-3
            neg_num = pos_num_TR - pos_num
            # score map loss
            singel_labels = flatten(Labels[:, :, :, 0:1])
            one_hot_labels = tf.concat([1 - singel_labels, singel_labels], axis=-1)
            loss = tf.losses.softmax_cross_entropy(one_hot_labels,
                                                   flatten(prediction[:, :, :, 0:2]),
                                                   reduction=tf.losses.Reduction.NONE)

            pos_flatten = tf.reshape(flatten(pos), shape=(-1,))
            pos_loss = loss * pos_flatten * pos_flatten_TR * Loss_weights
            neg_loss = loss * (1 - pos_flatten) * pos_flatten_TR * Loss_weights
            # neg_loss=tf.nn.top_k(neg_losses,k=neg_num).values
            score_loss = (tf.reduce_sum(pos_loss) / pos_num +
                          tf.reduce_sum(neg_loss) / neg_num)  # top_k not in use TODO: rebalance the pos/neg number !

        with tf.name_scope('Geo_loss'):
            geo_attr = ['radius', 'cos', 'sin']
            geo_loss = []
            total_loss = score_loss + TR_score_loss  # for training
            for i in range(4 - 1):  # self.configs.Label_size[2]-1):
                geo_loss.append(smooth_l1_loss(flatten(Labels[:, :, :, i + 1:i + 2] * pos * Loss_weights),
                                               flatten(prediction[:, :, :, i + 2:i + 3] * pos * Loss_weights)
                                               ) / pos_num)
                total_loss += geo_loss[-1]

        return total_loss, score_loss, geo_loss, geo_attr, TR_score_loss
