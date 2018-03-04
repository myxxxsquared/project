# this script is no longer in use


# assume that the file names are:
# id.npy
# id_maps.npy
# where id is a 8-char long str indicating its id

import glob
import os
import tqdm
from PIL import Image
import numpy as np


def ReadImg(path):
    """
    :param path: abs path of the img file
    :return:     tensor for the img
    """
    im = Image.open(path)  # shape=(h,w,c)
    return np.cast['float32'](np.array(im))  # covert unit8 to float32


def resize(im, resize):  # resize=(h,w,c)
    im = im.resize(resize[:2], Image.ANTIALIAS)
    return np.array(im)  # by default the image has been resized.


def Img_name(id):
    return id + '.npy'


def Labels_name(id):
    return id + '_maps.npy'

# the 2 functions above are not used in the new model


class raw_input(object):
    def __init__(self, configs, log):  # path/train, path/dev, path/test; should also include synthtext for pre-training
        train_path = os.path.join(configs.raw_data_path, 'Train')
        test_path = os.path.join(configs.raw_data_path, 'Test')
        SynthPath = configs.SynthPath

        self.train_path = os.path.join(configs.tfrecord_path, configs.dataset_name + '_train.tfrecords')
        self.test_path = os.path.join(configs.tfrecord_path, configs.dataset_name + '_test.tfrecords')
        self.SynthPath = os.path.join(configs.tfrecord_path, 'SynthText.tfrecords')

        if not os.path.isfile(self.train_path):
            log['info']('loading training data')
            self.train = self._load(train_path, log)
            log['info']('training data loaded')

        if not os.path.isfile(self.test_path):
            log['info']('loading test data')
            self.test = self._load(test_path, log)
            log['info']('test data loaded')

        if len(SynthPath) > 0 and (not os.path.isfile(self.SynthPath)):
            log['info']('loading SynthText data for pre-training')
            self.SynthText = self._load(SynthPath, log)
            log['info']('SynthText data loaded')
        else:
            self.SynthText = None

        log['info']('dataset loaded.')

    def _load(self, path, log):
        labels_files = glob.glob(os.path.join(path, Labels_name('*')))
        img_files = glob.glob(os.path.join(path, Img_name('*')))
        img_files = list(filter(lambda x: x.find('_maps') < 0, img_files))

        log['info']('Start to check input files.')
        ids = set(map(lambda x: x[:x.find('.npy')], img_files)) | set(map(lambda x: x[:x.find('_maps.npy')], labels_files))
        ids = list(ids)[:100000]  # trial
        length = len(ids)
        for id in tqdm.tqdm(range(length - 1, -1, -1)):
            if (Img_name(ids[id]) not in img_files)or(Labels_name(ids[id]) not in labels_files):
                log['debug']('Input files missing.')
                ids.pop(id)
                if (Img_name(ids[id]) not in img_files):
                    os.system('rm %s' % Img_name(ids[id]))
                if (Labels_name(ids[id]) not in labels_files):
                    os.system('rm %s' % Labels_name(ids[id]))
                #raise IOError('Input files missing.')
        log['info']('Valid Samples: %d.' % len(ids))

        data = []

        for id_num, id in enumerate(ids):
            data_instance = {'id': int(id.split('/')[-1])}
            data_instance['image'] = np.cast['float32'](np.load(Img_name(id)))
            data_instance['labels'] = np.load(Labels_name(id))
            data.append(data_instance)

        log['info']('Data loaded.')
        return data
