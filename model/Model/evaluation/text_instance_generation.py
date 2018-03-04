"""
This script demostrates the first version of our post-processing program.
It runs in the most primitive way and takes around 0.125s to reconstruct
text instances from a 128*128 feature map.

However as it takes much more time for larger scales, this script has been
modified and not used in our pipeline now. Please check the evaluate.py
under the directory 'data_management'. The algorithm demonstrated here has
been integrated into it.
"""

import numpy as np
from PIL import Image
import math
import cv2


def npy2image(matrix, scale=1):
    handle = Image.fromarray(matrix * scale)
    return handle


def tcl_reconstruction(input_path):
    """
    TODO: replace with cv2 functions to accelerate it
    :param input_path:path for an npy file. shape: (:,:,5). currently arranged as [TCL,radius,cos,sin,TR]
    :return:image(an instance for each channel), List[single instance point sets], List[convex hull]
    it takes 0.125s to rebuild from one pics on average
    """
    def demo(ins_list):
        show(np.concatenate(ins_list, axis=0))

    def show(image_array):
        return  # comment it to show the reconstructed text instances
        image = Image.fromarray(image_array * 255)
        image.show()

    output = np.load(input_path)
    TCL = output[:, :, 0]  # probability calculated from 0/1 channel outputs
    radius = output[:, :, 1]
    cos = output[:, :, 2]  # not used in our algorithm
    sin = output[:, :, 3]  # not used in our algorithm
    TR = output[:, :, 4]  # probability calculated from 0/1 channel outputs
    masked_TCL = TCL * TR   # multiplication is faster and logical '&' here.

    show(TCL)
    show(TR)

    def apply_circle(image, x, y, r):
        """
        This function draws/applies a circle onto the image
        :param image: ndarray(h,w,c)
        :param x:  x coordinate
        :param y:  y coordinate
        :param r:  radius
        :return:
        """
        for i in range(max(0, math.floor(x - r)), min(image.shape[0], math.ceil(x + r)) + 1):
            for j in range(max(0, math.floor(y - r)), min(image.shape[1], math.ceil(y + r)) + 1):
                if (i - x)**2 + (y - j)**2 <= r**2:
                    image[i, j] = 1
        return image

    def find_region(x, y, size=masked_TCL.shape):
        """
        using BFS to retrieve the group of pixels that belong to the same text instance
        :param x: starting point
        :param y: starting point
        :param size: size of the image, used to demarcate the image ranges
        :return:
        """
        instance_map = np.zeros(shape=size, dtype=np.int32)
        queue = [(x, y)]
        direction = ((-1, -1),
                     (-1, 0),
                     (-1, 1),
                     (0, -1),
                     (0, 1),
                     (1, -1),
                     (1, 0),
                     (1, 1))
        masked_TCL[queue[-1]] = 0
        while len(queue) > 0:
            cur_point = queue.pop(0)
            instance_map = apply_circle(instance_map, *cur_point, radius[cur_point])
            for i in range(8):
                x_, y_ = cur_point[0] + direction[i][0], cur_point[1] + direction[i][1]
                if x_ < 0 or y_ < 0 or x_ >= size[0] or y_ >= size[1]:
                    continue
                if masked_TCL[x_, y_] == 1:
                    queue.append((x_, y_))
                    masked_TCL[x_, y_] = 0
        return instance_map, np.transpose(np.nonzero(instance_map))

    instance_list = []
    cnt_list = []
    convex_hulls = []

    for i in range(masked_TCL.shape[0]):
        for j in range(masked_TCL.shape[1]):
            if masked_TCL[i, j] == 1:
                instance, cnt = find_region(i, j)
                instance_list.append(instance)
                cnt_list.append(cnt)
                convex_hulls.append(cv2.convexHull(cnt))
    # print('found ',str(len(instance_list)))#for debug

    demo(instance_list)

    return np.stack(instance_list, axis=-1), cnt_list, convex_hulls
