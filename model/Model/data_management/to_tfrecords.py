# this script is no longer in use
import tensorflow as tf
import os


class to_tfr(object):
    def __init__(self, configs, logs):
        self.configs = configs
        self.train_path = os.path.join(self.configs.tfrecord_path, self.configs.dataset_name)
        self.test_path = os.path.join(self.configs.tfrecord_path, self.configs.dataset_name)
        self.SynthPath = os.path.join(self.configs.tfrecord_path, 'synthtext')
        self.logs = logs
        self.Sess = None

    def trans(self, raw_input):
        """
        :param raw_input:raw_input class from raw_input.py
        :return:
        """
        if not os.path.isfile(self.train_path):
            self.logs['info']('to_tfrecords: training data')
            self.saving(self.train_path, raw_input.train)
        if not os.path.isfile(self.test_path):
            self.logs['info']('to_tfrecords: test data')
            self.saving(self.test_path, raw_input.test)
        if raw_input.SynthText and (not os.path.isfile(self.SynthPath)):
            self.logs['info']('to_tfrecords: SynthText data')
            self.saving(self.SynthPath, raw_input.SynthText)

        self.logs['info']('All data transformed into tfrecords.')

    def saving(self, save_path, raw_data):
        """
        :param raw_input:List[data] from raw_input class from raw_input.py
            save_path: ..../xx.tfrecords
            data_instance[id],[image],[labels]
        :return:
        """
        writer = tf.python_io.TFRecordWriter(save_path)
        for data_instance in raw_data:
            features = {
                "id": tf.train.Feature(int64_list=tf.train.Int64List(value=[data_instance['id']])),
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_instance['image'].tobytes()])),
                'labels': tf.train.Feature(bytes_list=tf.train.BytesList(value=[data_instance['labels'].tobytes()]))
            }
            example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(example.SerializeToString())  # 序列化为字符串

        writer.close()
