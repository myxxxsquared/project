from . import data_management
from . import network
from . import configs
from . import logs
import os
import time
import tensorflow as tf
import numpy as np
import cv2
import pickle
import glob

__name__ = 'Model'


class Model(object):
    def __init__(self):
        self.configuration = configs.configs()
        self.log_record = logs.logging_init(self.configuration)
        self.data_manager = data_management.data_batch(configs=self.configuration, logs=self.log_record)
        self.log_record['info']('data ready.')
        if not self.configuration.data_generate_mode:
            model_dict = {'Base': network.network_multi_gpu,
                          'Ver1': network.network_output_128,
                          'Ver2': network.network_output_kennel_size_1,
                          'Ver3': network.network_output_down_to_stage_1_deconv_1,
                          'Ver4': network.network_output_256_down_to_stage_1,
                          'Ver5': network.network_pixellink_style_simplified_prediction_module,
                          'Ver6': network.network_TCL,
                          'Ver7': network.network_new_loss,
                          'Ver8': network.network_TCL_old_loss,
                          'Ver9':network.network_output_down_to_stage_1_deconv_1_kernel_1,
                          'Ver10':network.network_geo_balanced,
                          'Ver11':network.network_IB,
                          'Ver12':network.network_IB_TCL}
            self.model = model_dict[self.configuration.model](configs=self.configuration, logs=self.log_record)
        self.initialize_flag = False

    def _initialize_model(self):
        if self.configuration.restore_path:
            self.load()
        else:
            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))  # TODO GPU control
            self.sess.run(tf.global_variables_initializer())
            self.log_record['info']('model initialized.')

        sum_path = os.path.join(self.configuration.summary_path_directory, "Sum")
        if os.path.isdir(sum_path):
            os.system('rm -r %s' % sum_path)
        self.writer = tf.summary.FileWriter(sum_path, self.sess.graph)
        self.log_record['info']('Tensorboard ready.')
        self.log_record['info']('Training starts.')
        self.pre_model_loss = 100  # recording the loss of the best performing model

        with tf.device('/device:cpu:0'):
            # recording test loss
            self.test_loss = tf.placeholder(dtype=tf.float32, shape=(), name='Test_loss')
            self.test_loss_sum = tf.summary.scalar("test_loss", self.test_loss)
            # image summary for test
            self.reconstructed_map = tf.placeholder(dtype=tf.uint8, shape=(None, None, None, 3), name='reconstructed_image')
            self.original_image = tf.placeholder(dtype=tf.uint8, shape=(None, None, None, 3), name='original_image')
            self.reconstructed_map_sum = tf.summary.image("y_reconstructed_map", self.reconstructed_map)
            self.original_image_sum = tf.summary.image("y_original_image", self.original_image)
            self.image4viz = tf.summary.merge([self.reconstructed_map_sum, self.original_image_sum])

        self.initialize_flag = True
        try:
            self.total_steps = self.sess.run(self.model.global_step)
        except:
            self.total_steps = 0

    def _train_step(self, image_summary=False, image_attribute_summary=False):
        if image_attribute_summary:
            # returning the geo attributes as well
            self.total_loss, _, self.summary, self.summary_image, self.image_attributes_summary_op, self.total_steps = self.sess.run([
                self.model.total_loss,
                self.model.train_step,
                self.model.summary_op,
                self.model.image_summary_op,
                self.model.image_attributes_summary_op,
                self.model.global_step],
                feed_dict={self.model.input_image: self.img_batch,
                           self.model.Labels: self.Labels_batch,
                           self.model.LossWeights: self.Weights_batch
                           })
            self.losses.append(self.total_loss)
            self.writer.add_summary(self.summary, self.total_steps)
            self.writer.add_summary(self.summary_image, self.total_steps)
            self.writer.add_summary(self.image_attributes_summary_op, self.total_steps)
        elif image_summary:
            # writing the image into the sum files
            self.total_loss, _, self.summary, self.summary_image, self.total_steps = self.sess.run(
                [
                    self.model.total_loss,
                    self.model.train_step,
                    self.model.summary_op,
                    self.model.image_summary_op,
                    self.model.global_step],
                feed_dict={self.model.input_image: self.img_batch,
                           self.model.Labels: self.Labels_batch,
                           self.model.LossWeights: self.Weights_batch
                           })
            self.losses.append(self.total_loss)
            self.writer.add_summary(self.summary, self.total_steps)
            self.writer.add_summary(self.summary_image, self.total_steps)
        else:
            # only record losses
            self.total_loss, _, self.summary, self.total_steps = self.sess.run(
                [
                    self.model.total_loss,
                    self.model.train_step,
                    self.model.summary_op,
                    self.model.global_step],
                feed_dict={self.model.input_image: self.img_batch,
                           self.model.Labels: self.Labels_batch,
                           self.model.LossWeights: self.Weights_batch
                           })
            self.losses.append(self.total_loss)
            self.writer.add_summary(self.summary, self.total_steps)

    def _obtain_data(self):
        return self.train_data_generator.get()

    def train(self, dataset='train', eval_per_step=200):
        """
        :param dataset: 'train', or 'Synth'
        :param eval_per_step: frequncy of doing evaluation on test set
        :return:
        """
        if not self.configuration.data_generate_mode:
            if not self.initialize_flag:
                self._initialize_model()
            if self.configuration.evaluate_before_training or self.configuration.test_mode:
                cur_loss = self.evaluate(0)
                test_sum = self.sess.run(self.test_loss_sum, feed_dict={self.test_loss: cur_loss})
                self.writer.add_summary(test_sum, self.total_steps)
            if self.configuration.test_mode:
                return []
            self.log_record['info']('Start to train on %s' % dataset)

            # training data loader initializer
        self.train_data_generator = self.data_manager.data_gen_multiprocess(dataset)
        count = 0
        self.losses = []
        for epoch in range(self.configuration.epochs[dataset]):
            self.log_record['info']('epoch %d starts: ' % epoch)

            if self.configuration.data_generate_mode:
                for i in range(eval_per_step):
                    _ = self._obtain_data()
                    count += self.configuration.batch_size
                print('%d done.' % count)
                continue

            for i in range(eval_per_step):
                start = time.time()

                t0 = time.time()
                self.id_batch, self.img_batch, self.Labels_batch, _, self.Weights_batch = self._obtain_data()
                print('data: %.4f' % (time.time() - t0), end='， ')

                t0 = time.time()
                self._train_step(image_summary=(i % 10 == 0), image_attribute_summary=(i % 200 == 0))
                print('train: %.4f' % (time.time() - t0))

                print('step: %5d, loss: %.5f' % (self.total_steps, self.total_loss), ', time: ', time.time() - start)

            self.log_record['info'](
                "step: %d, average training loss: %.3f" % (self.total_steps, sum(self.losses) / len(self.losses)))
            self.losses = []
            cur_loss = self.evaluate(self.total_steps)
            test_sum = self.sess.run(self.test_loss_sum, feed_dict={self.test_loss: cur_loss})
            self.writer.add_summary(test_sum, self.total_steps)
            self.model.save(self.sess,
                            os.path.join(self.configuration.save_path, self.configuration.code_name + '.cntk'))

            if cur_loss < self.pre_model_loss:
                self.model.save(self.sess, os.path.join(self.configuration.save_path, self.configuration.code_name + '_best_%d.cntk'%self.total_steps))
                if os.path.isdir(self.configuration.test_path + '_best'):
                    os.system('rm -r %s_best' % self.configuration.test_path)
                os.system('cp -r %s %s' % (self.configuration.test_path, self.configuration.test_path + '_best'))
                self.pre_model_loss = cur_loss
        if not self.configuration.data_generate_mode:
            self.model.save(self.sess,
                        os.path.join(self.configuration.save_path, self.configuration.code_name + '_' + dataset + '.cntk'))

        return self.losses

    def evaluate(self, total_steps):
        losses = []
        count = 0
        self.log_record['info']('start to evaluate')
        totaltext_recalls, totaltext_precisions, pascal_recalls, pascal_precisions = [], [], [], []
        self.test_data_generator = self.data_manager.data_test_set_gen(self.configuration.real_test)

        # Warning: the order of the prediction map is [TCL, radius, cos_theta, sin_theta,TR]
        for id_batch, img_batch, Labels_batch, cnt_ground_truth, weights_batch in self.test_data_generator:
            count += 1
            total_loss, summary_image, summary_image_attr, prediction, \
                TR_TCL_GT_map, L_R_map, L_sin, L_cos, \
                TR_TCL_Predictied_map, P_R_map, P_sin, P_cos \
                = self.sess.run([
                    self.model.total_loss,
                    self.model.image_summary_op,
                    self.model.image_attributes_summary_op,
                    self.model.prediction,
                    self.model.TR_TCL_GT_map,
                    self.model.L_R_map,
                    self.model.L_sin,
                    self.model.L_cos,
                    self.model.TR_TCL_Predictied_map,
                    self.model.P_R_map,
                    self.model.P_sin,
                    self.model.P_cos],
                    feed_dict={self.model.input_image: img_batch,
                               self.model.Labels: Labels_batch})

            self.writer.add_summary(summary_image, total_steps)
            self.writer.add_summary(summary_image_attr, total_steps)

            if self.configuration.test_mode:
                for i in range(img_batch.shape[0]):
                    t_r, t_p, p_r, p_p, visualize_map, precise_M, recall_M = data_management.evaluate(img_batch[i, :, :, :],
                                                                                                      cnts=cnt_ground_truth[i],
                                                                                                      is_text_cnts=True,
                                                                                                      is_viz=False,
                                                                                                      maps=prediction[i,:,:,:],NewEval=self.configuration.NewEval)
                    totaltext_recalls.append(t_r)
                    totaltext_precisions.append(np.nan_to_num(t_p))
                    pascal_recalls.append(p_r)
                    pascal_precisions.append(np.nan_to_num(p_p))
                    test_image_sum = self.sess.run(self.image4viz,
                                                   feed_dict={self.reconstructed_map: np.stack([visualize_map], axis=0),
                                                              self.original_image: img_batch[i:i + 1, :, :, ::-1]})

                    self.writer.add_summary(test_image_sum, self.total_steps + count * self.configuration.batch_size + i)
                    cv2.imwrite(filename=os.path.join(self.configuration.test_path, '%s_%.3f_%.3f_%.3f_%.3f.jpg' %
                                                      (id_batch[i], t_r, t_p, p_r, p_p)),
                                img=np.concatenate([np.concatenate([img_batch[i, :, :, :],
                                                                    np.cast['uint8'](TR_TCL_GT_map[i, :, :, :][:, :, (2, 1, 0)] * 255),
                                                                    np.cast['uint8'](L_R_map[i, :, :, :] * 255),
                                                                    np.cast['uint8'](L_sin[i, :, :, :] * 255),
                                                                    np.cast['uint8'](L_cos[i, :, :, :] * 255)], axis=0),
                                                    np.concatenate([visualize_map[:, :, (2, 1, 0)],
                                                                    np.cast['uint8'](TR_TCL_Predictied_map[i, :, :, :][:, :, (2, 1, 0)] * 255),
                                                                    np.cast['uint8'](P_R_map[i, :, :, :] * 255),
                                                                    np.cast['uint8'](P_sin[i, :, :, :] * 255),
                                                                    np.cast['uint8'](P_cos[i, :, :, :] * 255)], axis=0)], axis=1))  # left: original, right: predicted
                    pickle.dump((precise_M, recall_M), open(os.path.join(self.configuration.test_path, '%s_matrix.bin' % id_batch[i]), 'wb'))
                    pickle.dump(cnt_ground_truth[i],open(os.path.join(self.configuration.test_path, '%s_cnt.bin' % id_batch[i]), 'wb'))
                   # x=input('check')
                try:
                    self.save_instance(id_batch, prediction)
                except:
                    pass
            else:
                totaltext_recalls += [0] * img_batch.shape[0]
                totaltext_precisions += [0] * img_batch.shape[0]
                pascal_recalls += [0] * img_batch.shape[0]
                pascal_precisions += [0] * img_batch.shape[0]
            losses.append(total_loss * img_batch.shape[0])

        if True or self.configuration.dataset_name == 'totaltext':
            P = sum(totaltext_precisions) / len(totaltext_precisions)
            R = sum(totaltext_recalls) / len(totaltext_recalls)
        else:
            P = sum(totaltext_precisions) / len(totaltext_precisions)
            R = sum(pascal_recalls) / len(pascal_precisions)
        if P + R < 1e-3:
            F = 0
        else:
            F = 2 * P * R / (P + R)

        avg_loss = sum(losses) / self.configuration.size['test']

        self.log_record['info']('average test loss: %.3f. P=%.3f, R=%.3f, F=%.3f' % (avg_loss, P, R, F))

        return avg_loss

    def save_instance(self, *args):
        id = args[0]
        predictions = args[1]
        for sample in range(len(id)):
            np.save(os.path.join(self.configuration.test_path, '%s_maps.npy' % (id[sample])), predictions[sample, :, :, :])

    def load(self):
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.sess.run(tf.global_variables_initializer())
        self.model.load(sess=self.sess, path=self.configuration.restore_path)
        self.log_record['info']('model loaded from %s.' % self.configuration.restore_path)

    def predict(self, paths, output_dir, suffix='jpg'):
        """
        :param paths:
        :return:
        """
        if not self.initialize_flag:
            self._initialize_model()
        files = glob.glob(os.path.join(paths, '*' + suffix))
        for file in files:
            img = cv2.imread(file)
            size = img.shape
            if max(img.shape[:2]) > 2048:
                img = np.resize(img, new_shape=(int(size[0] * 2048 / max(img.shape[:2])) // 32 * 32,
                                                int(size[1] * 2048 / max(img.shape[:2])) // 32 * 32,
                                                size[2])).astype(np.uint8)
            prediction = self.sess.run([
                self.model.prediction
            ],
                feed_dict={self.model.input_image: np.stack([img], axis=0),
                           })
            masks = data_management.evaluate(img,
                                             cnts=None,
                                             is_text_cnts=True,
                                             is_viz=False,
                                             maps=[np.squeeze(map_) for map_ in np.split(prediction[0, :, :, :], 7, axis=-1)])
            np.save(os.path.join(output_dir, file.split('/')[-1][:-(len(suffix))] + str(size) + 'bin'), masks)


if __name__ == '__main__':
    pass
