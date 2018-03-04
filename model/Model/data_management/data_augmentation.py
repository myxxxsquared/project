#   please refer to https://github.com/aleju/imgaug
import cv2
import copy
import glob
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from random import shuffle, randint, choice, random, seed
# the following three libraries only serve to test this module
from multiprocessing import Pool
from .data_labelling import data_churn
import time


class DataAugmentor(object):
    """
    all data_augmentation defined below should take input as :
    input_data:
               Dict{'img_name':str,   original_name
                'img':np.uint8,
                'contour':see below,
                'is_text_cnts':if the cnt represents a text line or a char box
                'chars':List: indicating the char that each char box corresponds to}
                contour:
                if is_text_cnts is True: list(ndarray), ndarray: dtype np.float32, shape [n, 1, 2], order(col, row)
                if is_text_cnts is False: list(list(ndarray), list(ndarray)), for [char_cnts, text_cnts]

    synthetext: no augment, char, False;
    others: augment, box

    while returning the same format.

    valuable data augmentation types:
        **crop, pad, flip,
        invert/add(overall/channel), add to hue and saturation, multiply,
        gaussian blur, average blur, median blur, bilateral blur,
        sharpen, emboss, edge detect,
        noise: guassian(overall/channel), dropout(pixel/channel, coarse,channel), salt&pepper
        norm: contrast
        gray scale
        **affine transformation#i should implement it myself, maybe

    it takes around 0.25~0.3s to generate one augmented image on average.
    50% are processed with affine transformation

    """

    def __init__(self):
        self._add_augmentation_list = [
            iaa.Add((-30, 30), per_channel=True),
            iaa.Add((-30, 30), per_channel=False),
            iaa.AddElementwise((-30, 30), per_channel=False),
            iaa.AddElementwise((-30, 30), per_channel=True),
            iaa.Invert(p=0.2, per_channel=True),
            iaa.Invert(p=0.2, per_channel=False),
            iaa.AddToHueAndSaturation((0, 80), True),
            iaa.Multiply((0.8, 1.2), per_channel=True),
            iaa.Multiply((0.8, 1.2), per_channel=False),
            iaa.MultiplyElementwise((0.8, 1.2), per_channel=True),
            iaa.MultiplyElementwise((0.8, 1.2), per_channel=False)
        ]

        self._blur_augmentation_list = [
            iaa.GaussianBlur((2, 3)),
            iaa.AverageBlur((2, 3)),
            iaa.MedianBlur((3, 5)),
            iaa.BilateralBlur((2, 3))
        ]

        self._noise_augmentation_list = [
            iaa.AdditiveGaussianNoise(0, (5, 20), per_channel=True),
            iaa.AdditiveGaussianNoise(0, (5, 20), per_channel=False),
            iaa.Dropout((0.05, 0.15), per_channel=False),
            iaa.Dropout((0.05, 0.15), per_channel=True),
            iaa.CoarseDropout((0.05, 0.15), size_percent=(0.65, 0.85))
        ]

        self._other_augmentation_list = [
            iaa.Sharpen((0.9, 0.11), (0.8, 1.2)),
            iaa.Emboss((0.9, 0.11), (0.3, 1.6)),
            iaa.EdgeDetect((0, 0.4)),
            iaa.Grayscale((0, 1))
        ]

        self.noise_list = self._add_augmentation_list +\
            self._blur_augmentation_list +\
            self._noise_augmentation_list +\
            self._other_augmentation_list

    def _get_seq(self, affine=False):
        if affine:
            affine_seq = [
                iaa.Affine(rotate=360 * random() - 180),
                iaa.Affine(shear=80 * np.sin(random() * np.pi / 2) - 40),
                iaa.Noop(),
                iaa.Noop()
            ]
            shuffle(affine_seq)
            if random() < 0.4:
                scale_seq = []
            elif random() < 0.7:
                scale_seq = [iaa.Affine(scale={"x": 1.3 * random() + 0.7, "y": 1.3 * random() + 0.7})
                             ]
            else:
                x_scale = random()
                y_scale = random()
                scale_seq = [iaa.Affine(scale={"x": 0.6 * x_scale + 0.4, "y": 0.6 * y_scale + 0.4}),
                             iaa.Affine(scale={"x": 1 / (0.6 * x_scale + 0.4), "y": 1 / (0.6 * y_scale + 0.4)})]
            return iaa.Sequential(affine_seq[:2] + scale_seq)
        else:
            shuffle(self.noise_list)
            selected = self.noise_list[:2] + [iaa.Noop(), iaa.Noop()]
            shuffle(selected)
            return iaa.Sequential(selected[:2])

    @staticmethod
    def __key_points(image_shape, point_list):
        """
        feed cnt and return ia.KeypointsOnImage object
        :param point_list: np.array size=(n,1,2)
               image_shape
        :return:
        """
        keypoint_list = []
        for i in range(point_list.shape[0]):
            keypoint_list.append(ia.Keypoint(x=point_list[i, 0, 0], y=point_list[i, 0, 1]))
        return ia.KeypointsOnImage(keypoint_list,
                                   shape=ia.quokka(size=image_shape[:2]))

    @staticmethod
    def __resize_512p(input_data):
        """
        resize the short side of images under 512P to 512
        :param input_data:
        :return:
        return scaled img
        """
        rate = 1
        test_size=512
        if input_data['img'].shape[0] > input_data['img'].shape[1]:
            if True:  # input_data['img'].shape[1] < 512:
                rate = test_size / input_data['img'].shape[1]
                seq = iaa.Sequential([
                    iaa.Scale({'height': "keep-aspect-ratio", 'width': test_size}, 'cubic')
                ])
                input_data['img'] = seq.augment_image(input_data['img'])
        else:
            if True:  # input_data['img'].shape[0] < 512:
                rate = test_size / input_data['img'].shape[0]
                seq = iaa.Sequential([
                    iaa.Scale({'height': test_size, 'width': "keep-aspect-ratio"}, 'cubic')
                ])
                input_data['img'] = seq.augment_image(input_data['img'])

        if DataAugmentor._is_synthtext(input_data):
            input_data['contour'] = [[np.cast['int32'](contour * rate) for contour in contours] for contours in
                                     input_data['contour']]
        else:
            input_data['contour'] = [np.cast['int32'](contour * rate) for contour in input_data['contour']]
        input_data['center_point'] = [(np.cast['int32'](point[0] * rate),
                                       np.cast['int32'](point[1] * rate)) for point in input_data['center_point']]
        return input_data

    @staticmethod
    def __resize_to_32x(input_data):
        """
        resize image to multiples of 32, only for test on multiple aspect ratios
        :param input_data:
        :return:
        return scaled img
        """
        rate = 1
        while min(input_data['img'].shape[:2])>736:
            test_size = 736
            if input_data['img'].shape[0] > input_data['img'].shape[1]:
                if True:  # input_data['img'].shape[1] < 512:
                    rate = test_size / input_data['img'].shape[1]
                    seq = iaa.Sequential([
                        iaa.Scale({'height': "keep-aspect-ratio", 'width': test_size}, 'cubic')
                    ])
                    input_data['img'] = seq.augment_image(input_data['img'])
            else:
                if True:  # input_data['img'].shape[0] < 512:
                    rate = test_size / input_data['img'].shape[0]
                    seq = iaa.Sequential([
                        iaa.Scale({'height': test_size, 'width': "keep-aspect-ratio"}, 'cubic')
                    ])
                    input_data['img'] = seq.augment_image(input_data['img'])

            if DataAugmentor._is_synthtext(input_data):
                input_data['contour'] = [[np.cast['int32'](contour * rate) for contour in contours] for contours in
                                         input_data['contour']]
            else:
                input_data['contour'] = [np.cast['int32'](contour * rate) for contour in input_data['contour']]
            input_data['center_point'] = [(np.cast['int32'](point[0] * rate),
                                           np.cast['int32'](point[1] * rate)) for point in input_data['center_point']]
        p_row = 0
        while True:
            if (input_data['img'].shape[0] + p_row) % 32 == 0:
                break
            p_row += 1
        p_col = 0
        while True:
            if (input_data['img'].shape[1] + p_col) % 32 == 0:
                break
            p_col += 1

        input_data['img'] = np.pad(input_data['img'], ((0, p_row), (0, p_col), (0, 0)), mode='constant')

        return input_data

    @staticmethod
    def __resize_to_512(input_data):
        """
        resize image to test_size*test_size, only for tests
        :param input_data:
        :return:
        return scaled img
        """
        test_size = 720

        if DataAugmentor._is_synthtext(input_data):
            input_data['contour'] = [[np.cast['int32'](np.stack([contour[:, :, 0] * 512 / input_data['img'].shape[0],
                                                                 contour[:, :, 1] * 512 / input_data['img'].shape[1]],
                                                                axis=-1))
                                      for contour in contours] for contours in input_data['contour']]
        else:
            input_data['contour'] = [np.cast['int32'](np.stack([contour[:, :, 0] * 512 / input_data['img'].shape[0],
                                                                contour[:, :, 1] * 512 / input_data['img'].shape[1]],
                                                               axis=-1)) for contour in input_data['contour']]
        input_data['center_point'] = [(np.cast['int32'](point[0] * 512 / input_data['img'].shape[0]),
                                       np.cast['int32'](point[1] * 512 / input_data['img'].shape[1]))
                                      for point in input_data['center_point']]

        seq = iaa.Sequential([
            iaa.Scale({'height': 512, 'width': 512}, 'cubic')
        ])
        input_data['img'] = seq.augment_image(input_data['img'])

        return input_data

    def _resize(self, input_data, mode='train'):
        if mode == 'train':
            return DataAugmentor.__resize_512p(input_data)
        elif mode == 'test':
            return DataAugmentor.__resize_to_512(input_data)
        elif mode == 'real_test':
            return DataAugmentor.__resize_to_32x(input_data)
        raise ValueError('Invalid resize mode given.')

    @staticmethod
    def _pad(input_data):
        """
        pad the original image with black margins, so that affine transformation doesn't
        make the image cross the boundary.
        :param input_data:
        :return:
        """
        h = input_data['img'].shape[0]
        w = input_data['img'].shape[1]
        max_size = max([int(np.sqrt(np.power(h, 2) + np.power(w, 2))),
                        int(w + h * np.cos(11 / 36)),
                        int(max(h, w) * 2)
                        ]) + 5

        up = (max_size - h) // 2
        down = max_size - up - h
        left = (max_size - w) // 2
        right = max_size - left - w

        input_data['img'] = np.pad(input_data['img'], ((up, down), (left, right), (0, 0)), mode='constant')

        if DataAugmentor._is_synthtext(input_data):
            input_data['contour'][0] = list(
                map(lambda x: np.stack([x[:, :, 0] + up, x[:, :, 1] + left], axis=-1),  # x: np.array(n,1,2)
                    input_data['contour'][0]))
            input_data['contour'][1] = list(
                map(lambda x: np.stack([x[:, :, 0] + up, x[:, :, 1] + left], axis=-1),  # x: np.array(n,1,2)
                    input_data['contour'][1]))
        else:
            input_data['contour'] = list(
                map(lambda x: np.stack([x[:, :, 0] + up, x[:, :, 1] + left], axis=-1),  # x: np.array(n,1,2)
                    input_data['contour']))

        input_data['center_point'] = list(
            map(lambda x: (x[0] + up, x[1] + left),
                input_data['center_point']))
        return input_data

    def _pixel_augmentation(self, input_data):
        """
        doing augmentations that do not make affine transformation, including:

        invert/add(overall/channel), add to hue and saturation, multiply,
        gaussian blur, average blur, median blur, bilateral blur,
        sharpen, emboss, edge detect,
        noise: guassian(overall/channel), dropout(pixel/channel, coarse,channel), salt&pepper
        norm: contrast
        gray scale

        return: augmented image

        """
        #input_data = copy.deepcopy(inputs)

        input_data['img'] = self._get_seq().augment_image(input_data['img'])
        return input_data

    def _affine_transformation(self, input_data, trans_rate=1, synth=False):
        """
        affine types include:
            1. scaling
            2. rotate
            3. shear
            4. aspect ratios
        doing affine transformation
        trans_rate: probability of doing affine transformation

        Steps:

        (1) scaling + aspect ratios: TypeA: enlarge or reduce only once; TypeB: reduce first and then enlarge (so that the image is blurred, like a low-resolution one)
        (2) rotate
        (3) shear

        :return:
        """
        #input_data = copy.deepcopy(inputs)

        if random() > 1 - trans_rate:
            # obatin the transformer
            transformer = self._get_seq(affine=True)

            # doing the affine transformation
            input_data['img'] = transformer.augment_image(input_data['img'])
            if DataAugmentor._is_synthtext(input_data):
                for t in range(2):
                    contour_len = [cnt.shape[0] for cnt in input_data['contour'][t]]
                    concatenated_cnt = np.concatenate(input_data['contour'][t], axis=0)
                    splitting_point = [sum(contour_len[:i + 1]) for i in range(len(contour_len))]
                    input_data['contour'][t] = transformer.augment_keypoints([
                        self.__key_points(image_shape=input_data['img'].shape, point_list=concatenated_cnt)
                    ])[0]

                    input_data['contour'][t] = [np.array([[int(keypoints.y),
                                                           int(keypoints.x)]])[:, ::-1]
                                                for keypoints in input_data['contour'][t].keypoints]
                    input_data['contour'][t] = np.stack(input_data['contour'][t], axis=0)
                    input_data['contour'][t] = np.split(input_data['contour'][t], splitting_point[:-1])
            else:
                contour_len = [cnt.shape[0] for cnt in input_data['contour']]
                concatenated_cnt = np.concatenate(input_data['contour'], axis=0)
                splitting_point = [sum(contour_len[:i + 1]) for i in range(len(contour_len))]
                input_data['contour'] = transformer.augment_keypoints([
                    self.__key_points(image_shape=input_data['img'].shape, point_list=concatenated_cnt)
                ])[0]

                input_data['contour'] = [np.array([[int(keypoints.y),
                                                    int(keypoints.x)]])[:, ::-1]
                                         for keypoints in input_data['contour'].keypoints]
                input_data['contour'] = np.stack(input_data['contour'], axis=0)
                input_data['contour'] = np.split(input_data['contour'], splitting_point[:-1])

            # mapping center points
            input_data['center_point'][0] = \
                transformer.augment_keypoints([
                    self.__key_points(image_shape=input_data['img'].shape,
                                      point_list=np.array(
                        [[list(input_data['center_point'])[0]]]
                    ))
                ])[0].keypoints[0]
            input_data['center_point'][0] = (int(input_data['center_point'][0].y),
                                             int(input_data['center_point'][0].x))[::-1]
            input_data['center_point'][1] = transformer.augment_keypoints([
                self.__key_points(image_shape=input_data['img'].shape,
                                  point_list=np.array(
                    [[list(input_data['center_point'])[1]]]
                ))
            ])[0].keypoints[0]
            input_data['center_point'][1] = (int(input_data['center_point'][1].y),
                                             int(input_data['center_point'][1].x))[::-1]

        input_data['img'] = np.transpose(input_data['img'], axes=[1, 0, 2])  # ???

        return input_data

    def _crop(self, input_data):
        """
        flip, pad, crop, as final states.
        :param input_data:
        :return: the center point from which cropping is performed
        """
        center1 = input_data['center_point'][0]
        center2 = input_data['center_point'][1]
        p = random()
        p1 = center1[0] + int(p * (center2[0] - center1[0]))
        p2 = center1[1] + int(p * (center2[1] - center1[1]))
        return p1, p2

    @staticmethod
    def _is_synthtext(ins):
        return (not ins['is_text_cnts'])

    def augment(self, input_data, augment_rate=100, trans_rate=1, test_mode=False, real_test=False):
        """
        :param input_data:
               Dict{'img_name':str,   original_name
                'img':np.uint8,
                'contour':List[the contour of each text instance],
                'type': 'char' or 'tl',
                'is_text_cnts':if the cnt represents a text line or a char box,
                'chars':List: indicating the char that each char box corresponds to}
                contour:
                if is_text_cnts is True: list(ndarray), ndarray: dtype np.float32, shape [n, 1, 2], order(col, row)
                if is_text_cnts is False: list(list(ndarray), list(ndarray)), for [char_cnts, text_cnts] --> synthtext
        :param augment_rate
        :param trans_rate
        :return:
        DictDict{'img_name':str,   original_name
                'img':np.uint8,
                'contour':List[the contour of each text instance],
                'type': 'char' or 'tl',
                'is_text_cnts':if the cnt represents a text line or a char box,
                'char',
                'left_top': tuple (x, y), x is row, y is col, please be careful about the order,
                'right_bottom': tuple (x, y), x is row, y is col}

        Augmentation Steps:

        for synthtext data:
            (1) resize the short side to 512pxl
            (2) pad black background
            (3) (optional) affine transformation
            (4) crop along the central line

        for other training data:
            (1) resize the short side to 512pxl
            (2) pad black background
            (3) affine transformation
            (4) add noises
            (5) crop along the central line

        for test data:
            (1) resize to 512pxl*512pxl

        for test data with original size:
            (1) resize the short side to multiples of 512pxl

        """
        seed(int(time.time() * 1000))
        # step 1: reversing the x,y order
        if DataAugmentor._is_synthtext(input_data):
            if len(input_data['contour'][0][0].shape) < 3:
                for i in range(2):
                    input_data['contour'][i] = [np.stack([cnt], axis=1) for cnt in input_data['contour'][i]]
                    input_data['contour'][i] = [cnt[:, :, ::-1] for cnt in input_data['contour'][i]]
        else:
            if len(input_data['contour'][0].shape) < 3:
                input_data['contour'] = [np.stack([cnt], axis=1) for cnt in input_data['contour']]
            input_data['contour'] = [cnt[:, :, ::-1] for cnt in input_data['contour']]

        # step 2: calculating the center line later for cropping
        if input_data['img'].shape[0] < input_data['img'].shape[1]:
            input_data['center_point'] = [(input_data['img'].shape[0] // 2, input_data['img'].shape[0] // 2),
                                          (input_data['img'].shape[0] // 2, input_data['img'].shape[1] - input_data['img'].shape[0] // 2)]
        else:
            input_data['center_point'] = [(input_data['img'].shape[1] // 2, input_data['img'].shape[1] // 2),
                                          (input_data['img'].shape[0] - input_data['img'].shape[1] // 2, input_data['img'].shape[1] // 2)]

        # step 3: resizing
        if real_test:
            input_data = self._resize(input_data, mode='real_test')
            input_data['contour'] = [cnt[:, :, ::-1] for cnt in input_data['contour']]
            return {
                **input_data,
                'left_top': (0, 0),
                'right_bottom': (input_data['img'].shape[0], input_data['img'].shape[1])
            }

        if test_mode:
            input_data = self._resize(input_data, mode='test')
            input_data['contour'] = [cnt[:, :, ::-1] for cnt in input_data['contour']]
            return {
                **input_data,
                'left_top': (0, 0),
                'right_bottom': (input_data['img'].shape[0], input_data['img'].shape[1])
            }

        input_data = self._resize(input_data, mode='train')

        # step 4-6: padding & augmentation & cropping
        if random() < (1.0 / augment_rate):
            # only crop, without augmentation
            center_point = self._crop(input_data)
            if DataAugmentor._is_synthtext(input_data):
                for i in range(2):
                    input_data['contour'][i] = [cnt[:, :, ::-1] for cnt in input_data['contour'][i]]
            else:
                input_data['contour'] = [cnt[:, :, ::-1] for cnt in input_data['contour']]
            x_top = max(0, center_point[0] - 256)
            y_left = max(0, center_point[1] - 256)
            return {
                **input_data,
                'left_top': (x_top, y_left),
                'right_bottom': (x_top + 512, y_left + 512)
            }

        else:
            # step 4: padding
            input_data = self._pad(input_data)
            input_data['img'] = np.transpose(input_data['img'], axes=[1, 0, 2])  # ？？？
            # step 5: augmentation
            if DataAugmentor._is_synthtext(input_data):
                transformed = self._affine_transformation(input_data, trans_rate=trans_rate,
                                                          synth=True)
            else:
                transformed = self._affine_transformation(self._pixel_augmentation(input_data),
                                                          trans_rate=trans_rate, synth=False)
            # step 6: cropping
            center_point = self._crop(transformed)
            if DataAugmentor._is_synthtext(input_data):
                for i in range(2):
                    transformed['contour'][i] = [cnt[:, :, ::-1] for cnt in transformed['contour'][i]]
            else:
                transformed['contour'] = [cnt[:, :, ::-1] for cnt in transformed['contour']]
            x_top = max(0, center_point[0] - 256)
            y_left = max(0, center_point[1] - 256)
            return {**transformed,
                    'left_top': (x_top, y_left),
                    'right_bottom': (x_top + 512, y_left + 512)
                    }

    @staticmethod
    def demo(input_data_, cropped=True):
        """
        :param input_data:
        :param cropped: show the cropped img or not
        :return:
        """
        input_data = copy.deepcopy(input_data_)
        input_data['contour'] = [cnt[:, :, ::-1] for cnt in input_data['contour']]

        for i in range(len(input_data['contour'])):
            for point in range(input_data['contour'][i].shape[0]):
                input_data['img'][input_data['contour'][i][point, 0, 0] - 5:input_data['contour'][i][point, 0, 0] + 5,
                                  input_data['contour'][i][point, 0, 1] - 5:input_data['contour'][i][point, 0, 1] + 5, :] \
                    = (0, 255, 255)

        input_data['img'][input_data['center_point'][0][0] - 5:input_data['center_point'][0][0] + 5,
                          input_data['center_point'][0][1] - 5:input_data['center_point'][0][1] + 5, :] = (255, 255, 0)
        input_data['img'][input_data['center_point'][1][0] - 5:input_data['center_point'][1][0] + 5,
                          input_data['center_point'][1][1] - 5:input_data['center_point'][1][1] + 5, :] = (255, 255, 0)
        input_data['img'][input_data['left_top'][0]:input_data['right_bottom'][0],
                          input_data['left_top'][1]:input_data['right_bottom'][1], :] += 30
        if cropped:
            cv2.imshow('show', input_data['img'])  # np.transpose(img, axes=[1, 0, 2]))
        else:
            cv2.imshow('show', input_data['img'][input_data['left_top'][0]:input_data['right_bottom'][0],
                                                 input_data['left_top'][1]:input_data['right_bottom'][1], :])  # np.transpose(img, axes=[1, 0, 2]))
        cv2.waitKey(1)


if __name__ == '__main__':
    # codes here are used to test the data augmentation module
    try:
        images = glob.glob('/Users/jyouhoukun/Documents/PKU/courses/学校/2017-2018/深度学习算法应用/大作业/Total-Text-Dataset-master/Images/Train/img%d.jpg' % (randint(0, 1255)))
        shuffle(images)
        image1 = cv2.imread(images[0])
        image2 = cv2.imread(images[0])  # the second image is to check the performance when using multi-processes
    except:
        image1 = (np.random.randn(800, 1000, 3) * 255).astype(np.uint8)
        image2 = (np.random.randn(800, 1000, 3) * 255).astype(np.uint8)
    DA = DataAugmentor()
    labelling = data_churn()

    def data_label(ins):
        return labelling._data_labeling(ins['img_name'], ins['img'],
                                        ins['contour'], ins['is_text_cnts'],
                                        ins['left_top'], ins['right_bottom'])

    def process(ins):
        #t = time.time()
        image = DA.augment(ins)
        #print('Aug: ', time.time() - t)
        #t = time.time()
        name, image_, maps, cnt = data_label(image)
        #print('label: ',time.time() - t)
        return image, name, image_, maps

    input_ = {
        'img_name': 'aaa',
        'img': image1,
        'contour': [np.cast['int32'](np.array([[[100, 200]], [[110, 220]], [[109, 201]], [[120, 300]], [[300, 600]], [[100, 400]], [[110, 600]], [[210, 600]], [[200, 400]], [[200, 200]]]))],
        'type': 'tl',
        'is_text_cnts': True
    }
    input_['contour'] = [cnt[:, :, ::-1] for cnt in input_['contour']]

    _ = input('enter to see demo:')

    total = 0  # record time
    i_ = 0     # counting how many times the augmentation runs
    while i_ < 50:
        t0 = time.time()
        result = [process(input_)]
        _ = input('enter to see the augmented image: ')
        DA.demo(result[0][0], True)
        _ = input('enter to see the cropped area: ')
        DA.demo(result[0][0], False)

        i_ += 1
        total += time.time() - t0
    print(total / i_)
