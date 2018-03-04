# this script is no longer in use
import numpy as np
import tensorflow as tf
import cv2
import os


class Reader(object):
    """
    this class reads from the (a single) tfrecord and return
    {'img_name':str,   original_name
        'img':np.uint8,
        'contour':List[np.array(the contour of each text instance), (n,1,2)], ---> np.array(num_TI, num_point, 1, 2),
        'center_point':[(x1,y1),(x2,y2)]
        'is_text_cnts': bool, true for cnts of boxes,
                            false for cnts of char}
    """

    def __init__(self):
        pass

    def reader(self, path):
        record_iterator = tf.python_io.tf_record_iterator(path=path)
        for string_record in record_iterator:
            example = tf.train.Example()
            example.ParseFromString(string_record)

            img_index = int(example.features.feature['img_index']
                            .int64_list
                            .value[0])
            img_string = (example.features.feature['img']
                          .bytes_list
                          .value[0])
            contour_string = (example.features.feature['contour']
                              .bytes_list
                              .value[0])
            img_row = int(example.features.feature['im_row']
                          .int64_list
                          .value[0])
            img_col = int(example.features.feature['im_col']
                          .int64_list
                          .value[0])
            cnt_num = int(example.features.feature['cnt_num']
                          .int64_list
                          .value[0])
            cnt_point_num_string = (example.features.feature['cnt_point_num']
                                    .bytes_list
                                    .value[0])
            cnt_point_max = int(example.features.feature['cnt_point_max']
                                .int64_list
                                .value[0])

            img_1d = np.fromstring(img_string, dtype=np.uint8)
            reconstructed_img = img_1d.reshape((img_row, img_col, -1))
            img = reconstructed_img
            cnt_point_num = np.fromstring(cnt_point_num_string, dtype=np.int64)

            contour_1d = np.fromstring(contour_string, dtype=np.float32)
            reconstructed_contour = contour_1d.reshape((cnt_num, cnt_point_max, 1, 2))
            contour = []
            for i in range(cnt_num):
                contour.append(reconstructed_contour[i, :cnt_point_num[i], :, :].astype(np.int32))
            yield {
                'img_name': img_index,
                'img': img,
                'contour': contour,
                'is_text_cnts': not(path.find('synthtext') >= 0)
            }

    def demo(self, instance, path):
        img = cv2.drawContours(instance['img'], instance['contour'], -1, (0, 255, 0), 4)
        cv2.imwrite(path, img)
        os.system('imagcat %s' % path)
