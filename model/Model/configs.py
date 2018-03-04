"""
implementation of the configuration

includes:
(1) all hyper-parameters
(2) controlling units

Please read the third arg of each flag.

"""

import tensorflow as tf
import os


def to_type(data):
    if data == 'True':
        return True
    if data == 'False':
        return False
    if data.find('.') >= 0:
        return float(data)
    return int(data)


class configs(object):
    def __init__(self):
        tf.app.flags.DEFINE_integer("batch_size", 16, "batch_size per gpu")
        tf.app.flags.DEFINE_string('raw_data_path',
                                   '/home/lsb/data_cleaned/pkl/',
                                   'where to load the raw data')
        tf.app.flags.DEFINE_string('dataset_name', 'totaltext', 'dataset_name')
        tf.app.flags.DEFINE_string('SynthPath',
                                   '/media/sda/eccv2018/data/pkl/result2',
                                   'where to load the raw synthtext data')
        tf.app.flags.DEFINE_string("epochs", '150,150',
                                   "the number of epochs that synthtext and the target dataset respectively run")
        tf.app.flags.DEFINE_string('dataset_size', '1255/300/526991',
                                   'the number of samples in: training set, total set of the target dataset, and synthtexxt')  # 833019
        # tf.app.flags.DEFINE_string('tfrecord_path', 'tfrecord', 'tfrecord_path') this has been abandoned
        # tf.app.flags.DEFINE_string('weights_path', '~/pre_trained', 'dir that contained the pretrained base net weights') this is in fact not in use.
        tf.app.flags.DEFINE_string("input_size", '512*512*3', "= image_size")
        tf.app.flags.DEFINE_string("Label_size", '512*512*5', "= size of the GT labels")
        tf.app.flags.DEFINE_string("padding", 'SAME', "what padding strategy is used in conv layers")
        tf.app.flags.DEFINE_string("pooling", 'max', "what pooling strategy is used in conv layers")
        tf.app.flags.DEFINE_string("basenet", 'vgg16', "which base net to use. 'vgg16', 'vgg19', 'resnet-18, 34, 50, 101, 152'")
        tf.app.flags.DEFINE_float("learning_rate", 0.001, "the initial learning rate")    # TODO: add a new param for decaying type/ratio
        tf.app.flags.DEFINE_string("save_path", 'model_backup/', "the dir where we save the model periodically")
        tf.app.flags.DEFINE_float("momentum", 0.9, "a param for optimizer (SGD)")
        tf.app.flags.DEFINE_string("log_path", '', "where to save the log file")
        tf.app.flags.DEFINE_string("restore_path", '', "if not blank, indicating the saved model to load")
        tf.app.flags.DEFINE_string("test_path", 'test_output', "saving the test output")
        tf.app.flags.DEFINE_string('record_path',
                                   '/home/lsb/runninglog',
                                   'the dir to hold all the record/log, etc.')
        tf.app.flags.DEFINE_string('code_name', 'debug', 'the code for the experiment, only used in the name of the log file')
        # tf.app.flags.DEFINE_boolean('data_aug', True, 'data_aug')                abandoned
        # tf.app.flags.DEFINE_boolean('data_aug_noise', True, 'data_aug_noise')    abandoned
        tf.app.flags.DEFINE_string("US_Params", '3 3 2 2 same ReLU', "params for the upsampling layers: (kernel_size1, kernel_size2, stride1, stride2, padding, acti)")
        tf.app.flags.DEFINE_string("predict_channels", '128 64 32 32', "number of channels in the prediction channels")
        tf.app.flags.DEFINE_string("upsampling", 'DeConv', "upsampling type")
        tf.app.flags.DEFINE_integer("Predict_stage", 4, "the number of stages in the prediction modules")
        tf.app.flags.DEFINE_string("optimizer", 'Adam', "name of the optimizer")  # currently only supports 'Adam', 'YF'
        tf.app.flags.DEFINE_string('gpu_list', '0,4', 'which gpus to be used')  # '0,1,2,3,4,...'
        # tf.app.flags.DEFINE_boolean('multi_gpu_switch', True, 'multi_gpu_switch') abandoned
        tf.app.flags.DEFINE_boolean('real_test', False, 'NowDeprecated//True: use the original size(resized to the nearest multiples of 32) and aspect ratio to test')
        tf.app.flags.DEFINE_boolean('evaluate_before_training', False, 'True: evaluate before training')
        tf.app.flags.DEFINE_boolean('print_on_screen', False, 'True: print all the log onto the screen')
        tf.app.flags.DEFINE_boolean('use_pre_generated_data', True, 'True: use offline data')
        tf.app.flags.DEFINE_boolean('test_mode', False, 'True: do reconstruction to calculate P/R/F every test time')
        tf.app.flags.DEFINE_boolean('data_generate_mode', False, 'True: only generate offline data, do not train')
        tf.app.flags.DEFINE_integer("threads", 10, "number of threads/processes in generating/loading data")
        tf.app.flags.DEFINE_integer('QueueLength', 8, 'the max length of the queue to hold the data from data-generating processes')
        tf.app.flags.DEFINE_string('pre_labelled_data_path', '/home/lsb/pre_labelled_data', 'the dir to save the offline data')
        tf.app.flags.DEFINE_string('decay_type', 'expo',
                                   'the type of decaying learning rate')  # see Models.py --> network_multi_gpu --> _learning_rate
        tf.app.flags.DEFINE_string('decay_params', '5000/0.8/True',
                                   'the params of decaying learning rate')
        tf.app.flags.DEFINE_float("moving_average_decay", 0.99, "moving_average_decay")  # 0.99
        tf.app.flags.DEFINE_string('model', 'Base',
                                   'which mode to use')  # see Model/__init__.py Model->__init__
        tf.app.flags.DEFINE_boolean('prediction_mode', False,
                                    'make prediction')
        tf.app.flags.DEFINE_boolean('NewEval', False,
                                    'use the revised eval protocal or not')
        tf.app.flags.DEFINE_string('input_path', '',
                                   '')
        tf.app.flags.DEFINE_string('output_path', '',
                                   '')
        tf.app.flags.DEFINE_string('suffix', 'jpg',
                                   '')

        self.FLAGS = tf.app.flags.FLAGS
        self._read()

    def _read(self):
        self.record_path = self.FLAGS.record_path
        self.SynthPath = self.FLAGS.SynthPath
        self.dataset_name = self.FLAGS.dataset_name
        self.raw_data_path = self.FLAGS.raw_data_path
        self.epochs = list(map(int, self.FLAGS.epochs.split(',')))
        self.epochs = {'Synth': self.epochs[0], 'train': self.epochs[1]}
        self.input_size = self.image_size = tuple(map(int, self.FLAGS.input_size.split('*')))
        self.Label_size = tuple(map(int, self.FLAGS.Label_size.split('*')))
        self.padding = self.FLAGS.padding
        self.basenet = self.FLAGS.basenet
        self.pooling = self.FLAGS.pooling
        self.learning_rate = self.FLAGS.learning_rate
        self.momentum = self.FLAGS.momentum
        self.dataset_size = list(map(int, self.FLAGS.dataset_size.split('/')))
        self.size = {'train': self.dataset_size[0], 'test': self.dataset_size[1], "Synth": self.dataset_size[2]}
        self.save_path = os.path.join(self.record_path, self.FLAGS.save_path)
        self.log_path = os.path.join(self.record_path, self.FLAGS.log_path)
        self.restore_path = self.FLAGS.restore_path
        self.summary_path_directory = os.path.join('/media/sda/eccv2018/Sum', self.record_path.split('/')[-1])
        self.test_path = os.path.join(self.record_path, self.FLAGS.test_path)
        self.code_name = self.FLAGS.code_name
        self.US_Params = self.FLAGS.US_Params
        self.upsampling = self.FLAGS.upsampling
        self.predict_channels = list(map(int, self.FLAGS.predict_channels.split()))
        self.optimizer = self.FLAGS.optimizer
        self.Predict_stage = self.FLAGS.Predict_stage
        self.gpu_list = self.FLAGS.gpu_list.split(',')
        self.real_test = self.FLAGS.real_test
        self.NewEval = self.FLAGS.NewEval
        self.print_on_screen = self.FLAGS.print_on_screen
        self.threads = self.FLAGS.threads
        self.evaluate_before_training = self.FLAGS.evaluate_before_training
        self.QueueLength = self.FLAGS.QueueLength
        self.pre_labelled_data_path = self.FLAGS.pre_labelled_data_path
        self.use_pre_generated_data = self.FLAGS.use_pre_generated_data
        self.test_mode = self.FLAGS.test_mode
        self.data_generate_mode = self.FLAGS.data_generate_mode
        self.decay_type = self.FLAGS.decay_type
        self.decay_params = list(map(to_type, self.FLAGS.decay_params.split('/')))
        self.moving_average_decay = self.FLAGS.moving_average_decay
        self.model = self.FLAGS.model
        self.prediction_mode = self.FLAGS.prediction_mode
        self.input_path = self.FLAGS.input_path
        self.output_path = self.FLAGS.output_path
        if self.prediction_mode and (not os.path.isdir(self.output_path)):
            os.mkdir(self.output_path)
        self.suffix = self.FLAGS.suffix
        if self.prediction_mode and ((not os.path.isdir(self.input_path))or(not os.path.isdir(self.output_path))):
            raise ValueError('You should assign a input/output path.')
        os.environ["CUDA_VISIBLE_DEVICES"] = self.FLAGS.gpu_list
        self.batch_size = self.FLAGS.batch_size * len(self.gpu_list)
        for path in [self.record_path, self.save_path, self.log_path, self.summary_path_directory,
                     self.test_path]:
            if not os.path.isdir(path):
                os.makedirs(path)
